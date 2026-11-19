// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/macro_turing.h"
#if defined(ARCHITECTURE_x86_64) || defined(ARCHITECTURE_arm64)
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#else
#include "video_core/macro_turing_jit_a64.h"
#endif

// The JIT against the interpreter on small programs that cover every ALU operation, the flow
// instructions, the predicate, the outputs, the data RAM and the two callbacks: same emits, same
// consumed count, same data RAM, same success and error text.
namespace {

struct Lane {
    u32 op = 0, dst = 24, src_a = 24, src_b = 24, imm = 0;
};
struct Insn {
    std::array<Lane, 2> lane{};
    u32 pred_mode = 0, pred_reg = 0;
    std::array<u32, 2> out_method{0, 0}, out_emit{0, 0};
    bool end = false;
};

// Field offsets as TuringMacroInterpreter::Decode reads them.
std::array<u32, 3> Encode(const Insn& in) {
    std::array<u32, 3> bits{};
    const auto field = [&](u32 offset, u32 count, u32 value) {
        for (u32 bit = 0; bit < count; ++bit) {
            bits[(offset + bit) / 32] |= ((value >> bit) & 1) << ((offset + bit) % 32);
        }
    };
    field(0, 1, in.end ? 1 : 0);
    field(1, 4, in.pred_mode);
    field(5, 5, in.pred_reg);
    for (u32 l = 0; l < 2; ++l) {
        const u32 base = l * 36;
        field(10 + base, 5, in.lane[l].op);
        field(15 + base, 5, in.lane[l].dst);
        field(20 + base, 5, in.lane[l].src_a);
        field(25 + base, 5, in.lane[l].src_b);
        field(30 + base, 16, in.lane[l].imm);
    }
    field(82, 3, in.out_method[0]);
    field(85, 4, in.out_emit[0]);
    field(89, 3, in.out_method[1]);
    field(92, 4, in.out_emit[1]);
    return {bits[2], bits[1], bits[0]};
}

std::vector<u32> Program(const std::vector<Insn>& insns) {
    std::vector<u32> code;
    for (const Insn& in : insns) {
        const auto words = Encode(in);
        code.insert(code.end(), words.begin(), words.end());
    }
    // Trailing zero words, as the uploaded macro memory has.
    code.resize(code.size() + 6, 0);
    return code;
}

// Register 0 (the zero source) is 24; an immediate as a source is 25 (this lane's) / 26 (the
// other lane's); 27 is both immediates as one word; 28/29 the FIFO loads.
constexpr u32 Zero = 24, Imm = 25, ImmOther = 26, Imm32 = 27, Load0 = 28, Load1 = 29;

// An emit of lane 0's result to method `method` (output 5 = imm0 as the method word, with the
// increment in its upper bits) and output 1 = ALU 0.
Insn EmitAlu(u32 op, u32 dst, u32 src_a, u32 src_b, u32 method, u32 imm = 0, bool end = false) {
    Insn in;
    in.lane[0] = {op, dst, src_a, src_b, imm};
    in.lane[1] = {0, 24, 24, 24, (method & 0xfff) | (1u << 12)};
    in.out_method[1] = 6; // imm1: the method and increment 1
    in.out_emit[1] = 1;   // ALU 0's value
    in.end = end;
    return in;
}

struct Run {
    bool success{};
    std::string error;
    std::size_t consumed{};
    std::vector<std::pair<u32, u32>> emits;
    std::vector<u32> dram;
    std::vector<u32> reads;
};

Run RunInterpreter(std::span<const u32> code, u32 start, std::vector<u32> params,
                   std::vector<u32> dram, u32 refuse_method) {
    Tegra::TuringMacroInterpreter interpreter;
    Run run;
    run.dram = dram;
    std::copy(run.dram.begin(), run.dram.end(), interpreter.dram.begin());
    const auto result = interpreter.Execute(
        code, start, params,
        [&](u32 address) {
            run.reads.push_back(address);
            return address * 7 + 1;
        },
        [&](u32 method, u32 value) {
            run.emits.emplace_back(method, value);
            return method != refuse_method;
        });
    run.success = result.success;
    run.error = std::string{result.error};
    run.consumed = result.consumed;
    std::copy(interpreter.dram.begin(), interpreter.dram.begin() + run.dram.size(),
              run.dram.begin());
    return run;
}

Run RunJit(std::span<const u32> code, u32 start, std::vector<u32> params, std::vector<u32> dram,
           u32 refuse_method, bool batch) {
    Tegra::TuringMacroJit jit;
    jit.Prepare(code, 1, batch);
    Run run;
    std::vector<u32> full_dram(0xc00);
    std::copy(dram.begin(), dram.end(), full_dram.begin());
    Tegra::TuringMacroJit::Options options;
    options.batch_emits = batch;
    const auto result = jit.Execute(
        start, params, {}, full_dram,
        [&](u32 address) {
            run.reads.push_back(address);
            return address * 7 + 1;
        },
        [&](u32 method, u32 value) {
            run.emits.emplace_back(method, value);
            return method != refuse_method;
        },
        options);
    run.success = result.success;
    run.error = std::string{result.error};
    run.consumed = result.consumed;
    run.dram.assign(full_dram.begin(), full_dram.begin() + dram.size());
    return run;
}

void Same(const std::vector<Insn>& insns, u32 start, std::vector<u32> params,
          std::vector<u32> dram = std::vector<u32>(16, 0), u32 refuse_method = 0xfff) {
    const auto code = Program(insns);
    const Run expected = RunInterpreter(code, start, params, dram, refuse_method);
    for (const bool batch : {false, true}) {
        INFO("batch " << batch);
        const Run actual = RunJit(code, start, params, dram, refuse_method, batch);
        CHECK(actual.success == expected.success);
        CHECK(actual.error == expected.error);
        CHECK(actual.consumed == expected.consumed);
        CHECK(actual.emits == expected.emits);
        CHECK(actual.dram == expected.dram);
        CHECK(actual.reads == expected.reads);
    }
}

} // namespace

TEST_CASE("Turing macro JIT matches the interpreter: ALU operations", "[macro_jit]") {
    // R0 = p0, R1 = p1 (two FIFO loads), then every operation emitted to methods 0x100.. .
    std::vector<Insn> insns;
    insns.push_back(EmitAlu(0, 0, Load0, Zero, 0x100));  // R0 = p0 (+ 0), carry 0
    insns.push_back(EmitAlu(0, 1, Load0, Zero, 0x101));  // R1 = p1
    insns.push_back(EmitAlu(0, 2, 0, 1, 0x102));         // ADD, sets carry
    insns.push_back(EmitAlu(1, 3, 0, 1, 0x103));         // ADDC
    insns.push_back(EmitAlu(2, 4, 0, 1, 0x104));         // SUB, sets carry
    insns.push_back(EmitAlu(3, 5, 0, 1, 0x105));         // SUBB
    insns.push_back(EmitAlu(4, 6, 0, 1, 0x106));         // MUL signed, carry = high
    insns.push_back(EmitAlu(5, 7, Zero, Zero, 0x107));   // carry
    insns.push_back(EmitAlu(6, 8, 0, 1, 0x108));         // MUL unsigned
    insns.push_back(EmitAlu(5, 9, Zero, Zero, 0x109));
    insns.push_back(EmitAlu(8, 10, 0, Zero, 0x10a));     // CLZ
    insns.push_back(EmitAlu(9, 11, 0, 1, 0x10b));        // SLL
    insns.push_back(EmitAlu(10, 12, 0, 1, 0x10c));       // SRL
    insns.push_back(EmitAlu(11, 13, 0, 1, 0x10d));       // SRA
    insns.push_back(EmitAlu(12, 14, 0, 1, 0x10e));       // AND
    insns.push_back(EmitAlu(13, 15, 0, 1, 0x10f));       // NAND
    insns.push_back(EmitAlu(14, 16, 0, 1, 0x110));       // OR
    insns.push_back(EmitAlu(15, 17, 0, 1, 0x111));       // XOR
    // MERGE: dest 8, count 4, source 4 -> (8 << 10) | (4 << 5) | 4
    insns.push_back(EmitAlu(16, 18, 0, 1, 0x112, (8u << 10) | (4u << 5) | 4));
    for (u32 op = 17; op <= 21; ++op) {
        insns.push_back(EmitAlu(op, 19, 0, 1, 0x113 + (op - 17)));
    }
    insns.push_back(EmitAlu(0, 20, Imm, ImmOther, 0x120, 0x8001)); // immediates, sign-extended
    insns.push_back(EmitAlu(0, 21, Imm32, Zero, 0x121, 0x1234, true));
    for (const auto& [p0, p1] : std::vector<std::pair<u32, u32>>{
             {5, 3}, {3, 5}, {0xffffffff, 1}, {0x80000000, 0x7fffffff}, {0, 0}, {123456789, 987654321},
             {1u << 31, 33}, {0xf0f0f0f0, 0x0f0f0f0f}}) {
        INFO("p0 " << p0 << " p1 " << p1);
        Same(insns, 0, {p0, p1});
    }
}

TEST_CASE("Turing macro JIT matches the interpreter: loops, branches, END, stop", "[macro_jit]") {
    SECTION("a counted loop emits its counter three times") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(0, 0, Load0, Zero, 0x200)); // R0 = count
        // The interpreter alone refuses a read of a register never written (the JITs run with
        // zeroed registers, as the x86-64 one does), so the body's registers start written.
        insns.push_back(EmitAlu(0, 1, Zero, Zero, 0x205));
        insns.push_back(EmitAlu(0, 2, Zero, Zero, 0x206));
        Insn loop;
        loop.lane[0] = {23, 24, 0, 24, 3}; // LOOP R0 over the next 2 instructions
        insns.push_back(loop);
        insns.push_back(EmitAlu(0, 1, 1, Imm, 0x201, 1)); // R1 += 1, emit
        insns.push_back(EmitAlu(0, 2, 1, 1, 0x202));      // R2 = 2 * R1, emit
        insns.push_back(EmitAlu(0, 3, 1, Zero, 0x203, 0, true));
        insns.push_back(EmitAlu(0, 4, Zero, Zero, 0x204)); // the delay slot after END
        for (const u32 count : {0u, 1u, 3u, 7u}) {
            INFO("count " << count);
            Same(insns, 0, {count});
        }
    }
    SECTION("a branch forward and a branch back with the end cancel bit") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(0, 0, Load0, Zero, 0x300)); // R0 = p0
        Insn skip;                                          // if R0 == 0 skip the next emit
        skip.lane[0] = {21, 24, 0, 24, 0x8000 | 2};         // BEQ taken when equal, offset 2
        insns.push_back(skip);
        insns.push_back(EmitAlu(0, 1, Imm, Zero, 0x301, 7));
        insns.push_back(EmitAlu(2, 0, 0, Imm, 0x302, 1, true)); // R0 -= 1, END
        Insn back;                                           // delay slot: branch back while R0 != 0
        back.lane[0] = {21, 24, 0, 24, 0x4000 | (static_cast<u32>(-3) & 0x1fff)}; // not-equal, cancel END
        insns.push_back(back);
        insns.push_back(EmitAlu(0, 2, Imm, Zero, 0x303, 9));
        for (const u32 p0 : {0u, 1u, 3u}) {
            INFO("p0 " << p0);
            Same(insns, 0, {p0});
        }
    }
    SECTION("a jump and a stop") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(0, 0, Imm, Zero, 0x400, 1));
        Insn jump;
        jump.lane[0] = {24, 24, 24, 24, 2}; // JUMP to pc + 2
        insns.push_back(jump);
        insns.push_back(EmitAlu(0, 1, Imm, Zero, 0x401, 2)); // skipped
        insns.push_back(EmitAlu(0, 2, Imm, Zero, 0x402, 3));
        Insn stop;
        stop.lane[0] = {21, 24, 24, 24, 0x8000 | (static_cast<u32>(-4096) & 0x1fff)}; // equal: stop
        insns.push_back(stop);
        insns.push_back(EmitAlu(0, 3, Imm, Zero, 0x403, 4)); // the stop ends after this one
        insns.push_back(EmitAlu(0, 4, Imm, Zero, 0x404, 5)); // not run
        Same(insns, 0, {});
    }
    SECTION("a start in the middle and a start outside the code") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(0, 0, Imm, Zero, 0x500, 1));
        insns.push_back(EmitAlu(0, 1, Imm, Zero, 0x501, 2, true));
        insns.push_back(EmitAlu(0, 2, Imm, Zero, 0x502, 3));
        Same(insns, 1, {});
        Same(insns, 7, {});
    }
}

TEST_CASE("Turing macro JIT matches the interpreter: predicate, outputs, data RAM, reads", "[macro_jit]") {
    SECTION("predicated lanes and outputs") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(0, 0, Load0, Zero, 0x600));
        for (u32 mode = 1; mode < 16; ++mode) {
            Insn in;
            in.pred_mode = mode;
            in.pred_reg = 0;
            in.lane[0] = {0, 1, Imm, Zero, 10 + mode};
            in.lane[1] = {0, 2, Imm, Zero, (0x610u + mode) | (1u << 12)};
            in.out_method[0] = 6; // imm1 as the method
            in.out_emit[0] = 1;   // ALU 0
            in.out_method[1] = 6;
            in.out_emit[1] = 2;   // ALU 1
            insns.push_back(in);
        }
        insns.back().end = true;
        Same(insns, 0, {0});
        Same(insns, 0, {1});
    }
    SECTION("every output selector") {
        std::vector<Insn> insns;
        for (const u32 selector : {3u, 4u, 5u, 6u, 8u, 9u, 10u}) {
            Insn in;
            in.lane[0] = {0, 0, Load0, Load1, 0xabcd};
            in.lane[1] = {0, 1, Zero, Zero, 0x7001};
            in.out_method[0] = 6;
            in.out_emit[0] = selector;
            insns.push_back(in);
        }
        insns.back().end = true;
        Same(insns, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    }
    SECTION("data RAM writes and reads") {
        std::vector<Insn> insns;
        Insn write;
        write.lane[0] = {31, 24, Imm, ImmOther, 3};       // DWRITE [3] = imm1 (lane 1's 0x55)
        write.lane[1] = {0, 24, Zero, Zero, 0x55};
        insns.push_back(write);
        insns.push_back(EmitAlu(30, 0, Imm, Zero, 0x700, 3)); // DREAD [3] -> emit
        insns.push_back(EmitAlu(30, 1, Imm, Zero, 0x701, 5, true)); // DREAD [5]
        Same(insns, 0, {}, {0, 1, 2, 3, 4, 0x99, 6, 7});
    }
    SECTION("register reads through the callback") {
        std::vector<Insn> insns;
        insns.push_back(EmitAlu(22, 0, Imm, Zero, 0x800, 0x1234)); // READ 0x1234 -> emit
        insns.push_back(EmitAlu(22, 1, Imm, Imm, 0x801, 0x9000, true)); // READ (0x9000+0x9000) & 0xffff
        Same(insns, 0, {});
    }
    SECTION("the emits keep their order across batching and a refused method fails both") {
        std::vector<Insn> insns;
        for (u32 i = 0; i < 40; ++i) {
            insns.push_back(EmitAlu(0, 0, Imm, Zero, 0x100 + i, i));
        }
        insns.push_back(EmitAlu(22, 1, Imm, Zero, 0x900, 5)); // a read flushes the batch
        insns.push_back(EmitAlu(0, 2, Imm, Zero, 0xfff, 1, true)); // refused
        insns.push_back(EmitAlu(0, 3, Imm, Zero, 0x901, 1));
        Same(insns, 0, {});
    }
}

TEST_CASE("Turing macro JIT matches the interpreter: errors", "[macro_jit]") {
    SECTION("FIFO underflow") {
        Same({EmitAlu(0, 0, Load0, Load1, 0x100), EmitAlu(0, 1, Load0, Zero, 0x101, 0, true)}, 0,
             {1, 2});
    }
    SECTION("data RAM out of bounds") {
        Same({EmitAlu(30, 0, Imm, Zero, 0x100, 0xc00, true)}, 0, {});
        Insn write;
        write.lane[0] = {31, 24, Imm, Zero, 0xc01};
        write.end = true;
        Same({write}, 0, {});
    }
    SECTION("invalid loop and nested loop") {
        Insn loop;
        loop.lane[0] = {23, 24, Imm, 24, 1}; // imm < 2
        Same({loop}, 0, {});
        std::vector<Insn> nested;
        nested.push_back(EmitAlu(0, 0, Imm, Zero, 0x100, 2));
        Insn outer;
        outer.lane[0] = {23, 24, 0, 24, 4};
        nested.push_back(outer);
        Insn inner;
        inner.lane[0] = {23, 24, 0, 24, 2};
        nested.push_back(inner);
        nested.push_back(EmitAlu(0, 1, Imm, Zero, 0x101, 1));
        nested.push_back(EmitAlu(0, 2, Imm, Zero, 0x102, 1, true));
        Same(nested, 0, {});
    }
    SECTION("unsupported extended operation and invalid merge") {
        Same({EmitAlu(7, 0, Imm, Zero, 0x100, 0x1000, true)}, 0, {});
        Same({EmitAlu(7, 0, Imm, Zero, 0x100, 0x2000, true)}, 0, {});
        Same({EmitAlu(16, 0, Imm, Imm, 0x100, (40u << 10) | (4u << 5), true)}, 0, {});
    }
    SECTION("a macro that never ends hits the step limit") {
        Insn spin;
        spin.lane[0] = {24, 24, 24, 24, 0}; // JUMP to itself
        Same({spin}, 0, {});
    }
}
#endif
