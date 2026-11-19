// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>

#include "video_core/macro_turing.h"

namespace Tegra {

// AArch64 code for the TU104/GA10x MME, the arm64 twin of macro_turing_jit.h (x86-64): every
// instruction of the uploaded macro memory compiled once (each a block that falls through to the
// next one, branches and loops through a jump table), with the semantics of
// TuringMacroInterpreter::Execute. Emits (method writes) and register reads call back into C++.
// State, Options, Error and Execute are the same as the x86-64 class's and must stay in step with
// it: macro.cpp and maxwell_3d.cpp use whichever the architecture provides.
//
// Register use in the generated code: x19 = State*; w20/w21 = the instruction's FIFO loads;
// w22/w23 = its ALU results (all callee-saved, so the callbacks keep them); w8 = x, w9 = y,
// w10 = a lane's value, w11-w15 scratch, x16 the call target.
class TuringMacroJit {
public:
    using Word = std::uint32_t;
    using Result = TuringMacroInterpreter::Result;

    /// Emits a batching program buffers before calling `emit` for them in order.
    static constexpr Word EmitBatch = 32;

    struct State {
        std::array<Word, 24> regs;
        Word carry;
        Word method;
        Word increment;
        Word loop_count;
        Word loop_start;
        Word loop_end;
        Word next_pc;
        Word steps;
        Word start;
        Word consumed;
        Word params_size;
        Word fail_pc;
        std::uint8_t end_next;
        std::uint8_t stopped;
        std::uint8_t cancel_end;
        std::uint8_t pad;
        Word scratch_x;
        Word scratch_y;
        const Word* params;
        Word* dram;
        void* user;
        bool (*emit)(State*, Word method, Word value);
        Word (*read)(State*, Word address);
        const std::vector<Word>* fifo;
        Word emit_count;
        Word emit_capacity; ///< Emits buffered before `emit` runs; 0 calls it for each
        std::uint8_t borrowed; ///< `params` is not `fifo`'s until the FIFO holds words
        std::array<Word, 2 * EmitBatch> emit_buffer; ///< (method, value) pairs
    };

    struct Options {
        /// Buffer emits of methods below 0xe00 other than MME_DMA_READ(_FIFOED) (0x157/0x158)
        /// and call `emit` for them in order before a register read, before any other emit, when
        /// the buffer fills and at the end. `emit` must accept every method buffered. Needs a
        /// program prepared with batch_emits.
        bool batch_emits{};
        /// Reuse one State across calls instead of initializing every field.
        bool keep_state{};
    };

    enum Error : Word {
        None = 0,
        FifoUnderflow,
        InvalidSource,
        InvalidDestination,
        UnsupportedExtended,
        InvalidMerge,
        InvalidLoop,
        DataRamReadOutOfBounds,
        DataRamWriteOutOfBounds,
        InvalidOutput,
        UnsupportedEmit,
        OutsideCode,
        StepLimit,
    };

    static std::string_view ErrorText(Word error) {
        constexpr std::array<std::string_view, 13> texts{
            "",
            "parameter FIFO underflow",
            "invalid source register",
            "invalid destination register",
            "unsupported extended operation",
            "invalid merge destination",
            "invalid loop",
            "data RAM read out of bounds",
            "data RAM write out of bounds",
            "invalid output selector",
            "unsupported emitted method",
            "instruction outside uploaded RAM",
            "instruction limit exceeded",
        };
        return error < texts.size() ? texts[error] : "unknown";
    }

    /// Compiles `code` (3 words per instruction) when it differs from what was compiled.
    /// The plain and the batching program are kept apart: callers that share one engine's JIT
    /// (3D macros and compute's front) may want different ones on alternate calls.
    void Prepare(std::span<const Word> code, std::uint64_t generation, bool batch_emits = false) {
        const std::size_t slot = batch_emits ? 1 : 0;
        if (!programs[slot] || generation != compiled_generations[slot]) {
            programs[slot] = std::make_unique<Program>(code, batch_emits);
            compiled_generations[slot] = generation;
        }
        program = programs[slot].get();
        compiled_batch = batch_emits;
    }

    template <typename Read, typename Emit>
    Result Execute(Word start, std::vector<Word>& parameters, std::span<Word> data_ram, Read read,
                   Emit emit) {
        return Execute(start, parameters, {}, data_ram, read, emit, Options{});
    }

    /// `borrowed`: the parameters when they are not in `fifo` (which is empty then). An emit that
    /// appends to `fifo` copies them there first; the program reads from `fifo` from then on.
    template <typename Read, typename Emit>
    Result Execute(Word start, std::vector<Word>& fifo, std::span<const Word> borrowed,
                   std::span<Word> data_ram, Read read, Emit emit, Options options) {
        State local;
        const bool keep = options.keep_state && !kept_busy;
        if (keep && !kept) {
            kept = std::make_unique<State>();
        }
        State& state = keep ? *kept : local;
        state.regs.fill(0);
        state.carry = state.method = state.increment = 0;
        state.loop_count = 0;
        state.consumed = 0;
        state.end_next = state.stopped = 0;
        if (!keep) {
            // A kept State holds these from its last call: none is read before the call writes
            // it. cancel_end and next_pc are set first by every flow instruction, scratch_x/y
            // right before their use, loop_start/loop_end are read only while loop_count is
            // not 0 (only a LOOP sets it, writing both), and nothing writes fail_pc.
            state.loop_start = state.loop_end = 0;
            state.fail_pc = 0;
            state.cancel_end = 0;
        }
        state.start = start;
        state.steps = 1000000;
        const bool borrow = !borrowed.empty() && fifo.empty();
        state.params = borrow ? borrowed.data() : fifo.data();
        state.params_size = static_cast<Word>(borrow ? borrowed.size() : fifo.size());
        state.borrowed = borrow ? 1 : 0;
        state.dram = data_ram.data();
        state.fifo = &fifo;
        state.emit_count = 0;
        state.emit_capacity = options.batch_emits && compiled_batch ? EmitBatch : 0;
        Thunks<Read, Emit> thunks{read, emit};
        state.user = &thunks;
        state.emit = &Thunks<Read, Emit>::CallEmit;
        state.read = &Thunks<Read, Emit>::CallRead;
        kept_busy |= keep;
        const Word error = program->entry(&state);
        if (state.emit_count != 0) {
            Thunks<Read, Emit>::Flush(&state);
        }
        if (keep) {
            kept_busy = false;
        }
        if (error == None) {
            return {true, state.fail_pc, state.consumed, {}};
        }
        return {false, state.fail_pc, state.consumed, ErrorText(error)};
    }

private:
    template <typename Read, typename Emit>
    struct Thunks {
        Read& read;
        Emit& emit;
        static void Flush(State* state) {
            Thunks& self = *static_cast<Thunks*>(state->user);
            const Word count = state->emit_count;
            state->emit_count = 0;
            for (Word i = 0; i < count; ++i) {
                self.emit(state->emit_buffer[i * 2], state->emit_buffer[i * 2 + 1]);
            }
        }
        static bool CallEmit(State* state, Word method, Word value) {
            if (state->emit_count != 0) {
                Flush(state);
            }
            const bool ok = static_cast<Thunks*>(state->user)->emit(method, value);
            // MME_DMA_READ_FIFOED appends to the FIFO from inside the emit.
            if (!state->borrowed || !state->fifo->empty()) {
                state->params = state->fifo->data();
                state->params_size = static_cast<Word>(state->fifo->size());
            }
            return ok;
        }
        static Word CallRead(State* state, Word address) {
            if (state->emit_count != 0) {
                Flush(state);
            }
            return static_cast<Thunks*>(state->user)->read(address);
        }
    };

    struct Program {
        explicit Program(std::span<const Word> code, bool batch_emits_)
            : batch_emits{batch_emits_}, mem{CodeSize(Count(code))} {
            mem.unprotect();
            oaknut::CodeGenerator generator{mem.ptr()};
            Generator{*this, generator}.Generate(code);
            mem.protect();
            mem.invalidate_all();
            entry = reinterpret_cast<Word (*)(State*)>(mem.ptr());
        }

        Word (*entry)(State*) = nullptr;

    private:
        using Decoded = TuringMacroInterpreter::Decoded;

        // Every instruction of the uploaded memory is compiled, the zero words past the last
        // macro too: an END runs the instruction after it, and for the last macro in RAM that is
        // a zero word.
        static Word Count(std::span<const Word> code) {
            return static_cast<Word>(code.size() / 3);
        }

        // The longest instruction (two predicated bodies, both lanes with a callback and batched
        // emits in each) is under 200 AArch64 instructions; the entry, the stubs and the table
        // come on top.
        static std::size_t CodeSize(Word count) {
            return (std::size_t{count} * 256 + 1024) * sizeof(std::uint32_t) + std::size_t{count} * 8;
        }

        static constexpr std::size_t RegOffset(Word reg) {
            return offsetof(State, regs) + reg * sizeof(Word);
        }

        struct Generator {
            Program& program;
            oaknut::CodeGenerator& code;
            using Cond = oaknut::Cond;
            using Label = oaknut::Label;

            // The callee-saved registers the generated code keeps across callbacks.
            const oaknut::XReg st = oaknut::util::X19;
            const oaknut::WReg load0 = oaknut::util::W20, load1 = oaknut::util::W21;
            const oaknut::WReg alu0 = oaknut::util::W22, alu1 = oaknut::util::W23;
            // Caller-saved scratch.
            const oaknut::WReg x = oaknut::util::W8, y = oaknut::util::W9, value = oaknut::util::W10;
            const oaknut::WReg t0 = oaknut::util::W11, t1 = oaknut::util::W12, t2 = oaknut::util::W13;
            const oaknut::XReg xt0 = oaknut::util::X11, xt1 = oaknut::util::X12, xt2 = oaknut::util::X13,
                               xt3 = oaknut::util::X14, call = oaknut::util::X16, xvalue = oaknut::util::X10;

            Label table, dispatch, success, leave;
            Label fail_underflow, fail_source, fail_destination, fail_extended, fail_merge,
                fail_loop, fail_dread, fail_dwrite, fail_output, fail_emit, fail_outside,
                fail_steps;
            std::vector<Label> labels;
            std::vector<bool> loop_end, after_end;

            // Conditional branches reach 1 MiB; the exits sit at the start of a program that can
            // be larger, so a far conditional branch inverts its condition over a plain B.
            void BranchFar(Cond cond, Label& target) {
                Label skip;
                code.B(oaknut::invert(cond), skip);
                code.B(target);
                code.l(skip);
            }
            void CbzFar(oaknut::WReg reg, Label& target) {
                Label skip;
                code.CBNZ(reg, skip);
                code.B(target);
                code.l(skip);
            }
            void CbnzFar(oaknut::WReg reg, Label& target) {
                Label skip;
                code.CBZ(reg, skip);
                code.B(target);
                code.l(skip);
            }
            void LoadField(oaknut::WReg dst, std::size_t offset) {
                code.LDR(dst, st, static_cast<std::uint32_t>(offset));
            }
            void StoreField(oaknut::WReg src, std::size_t offset) {
                code.STR(src, st, static_cast<std::uint32_t>(offset));
            }
            void LoadByte(oaknut::WReg dst, std::size_t offset) {
                code.LDRB(dst, st, static_cast<std::uint32_t>(offset));
            }
            void StoreByte(oaknut::WReg src, std::size_t offset) {
                code.STRB(src, st, static_cast<std::uint32_t>(offset));
            }
            void StoreByteImm(Word imm, std::size_t offset) {
                code.MOV(t0, imm);
                StoreByte(t0, offset);
            }
            void StoreFieldImm(Word imm, std::size_t offset) {
                code.MOV(t0, imm);
                StoreField(t0, offset);
            }
            // The table and the dispatch code are at the start: every block reaches them with a
            // plain B (128 MiB), the entry reaches the table with ADR (1 MiB).
            void JumpTable(oaknut::WReg pc_reg) {
                code.ADR(xt0, table);
                code.LDR(xt1, xt0, pc_reg, oaknut::IndexExt::UXTW, 3);
                code.BR(xt1);
            }

            void Generate(std::span<const Word> full_code) {
                const Word count = Count(full_code);
                const std::span<const Word> code_span = full_code.first(std::size_t{count} * 3);
                labels.resize(count);
                std::vector<Decoded> decoded(count);
                for (Word pc = 0; pc < count; ++pc) {
                    decoded[pc] = TuringMacroInterpreter::Decode(
                        {code_span[pc * 3 + 2], code_span[pc * 3 + 1], code_span[pc * 3]});
                }
                Analyze(decoded);

                // Prologue: x29/x30 and x19-x23 saved, 64 bytes, the stack stays 16-aligned.
                code.STP(oaknut::util::X29, oaknut::util::X30, oaknut::util::SP, oaknut::PreIndexed{}, -64);
                code.STP(oaknut::util::X19, oaknut::util::X20, oaknut::util::SP, 16);
                code.STP(oaknut::util::X21, oaknut::util::X22, oaknut::util::SP, 32);
                code.STR(oaknut::util::X23, oaknut::util::SP, 48);
                code.MOV(st, oaknut::util::X0);
                LoadField(value, offsetof(State, start));
                code.MOV(t0, count);
                code.CMP(value, t0);
                code.B(Cond::HS, fail_outside);
                JumpTable(value);

                // Dispatch on w10 (the next pc). Every backward edge comes through here, so the
                // instruction limit is counted here.
                code.l(dispatch);
                LoadField(t0, offsetof(State, steps));
                code.SUBS(t0, t0, 1);
                code.B(Cond::EQ, fail_steps);
                StoreField(t0, offsetof(State, steps));
                code.MOV(t0, count);
                code.CMP(value, t0);
                code.B(Cond::HS, fail_outside);
                JumpTable(value);

                code.l(success);
                code.MOV(oaknut::util::W0, oaknut::util::WZR);
                code.l(leave);
                code.LDR(oaknut::util::X23, oaknut::util::SP, 48);
                code.LDP(oaknut::util::X21, oaknut::util::X22, oaknut::util::SP, 32);
                code.LDP(oaknut::util::X19, oaknut::util::X20, oaknut::util::SP, 16);
                code.LDP(oaknut::util::X29, oaknut::util::X30, oaknut::util::SP, oaknut::PostIndexed{}, 64);
                code.RET();
                const auto fail_with = [&](Label& label, Word error) {
                    code.l(label);
                    code.MOV(oaknut::util::W0, error);
                    code.B(leave);
                };
                fail_with(fail_underflow, FifoUnderflow);
                fail_with(fail_source, InvalidSource);
                fail_with(fail_destination, InvalidDestination);
                fail_with(fail_extended, UnsupportedExtended);
                fail_with(fail_merge, InvalidMerge);
                fail_with(fail_loop, InvalidLoop);
                fail_with(fail_dread, DataRamReadOutOfBounds);
                fail_with(fail_dwrite, DataRamWriteOutOfBounds);
                fail_with(fail_output, InvalidOutput);
                fail_with(fail_emit, UnsupportedEmit);
                fail_with(fail_outside, OutsideCode);
                fail_with(fail_steps, StepLimit);

                // The table's entries are filled once the blocks' addresses are known.
                code.align(8);
                code.l(table);
                std::uint64_t* const table_entries = code.xptr<std::uint64_t*>();
                for (Word pc = 0; pc < count; ++pc) {
                    code.dx(0);
                }

                std::vector<std::uintptr_t> block_addresses(count);
                for (Word pc = 0; pc < count; ++pc) {
                    code.l(labels[pc]);
                    block_addresses[pc] = code.xptr<std::uintptr_t>();
                    Instruction(pc, count, decoded[pc]);
                }
                for (Word pc = 0; pc < count; ++pc) {
                    table_entries[pc] = block_addresses[pc];
                }
            }

            // Loads MME register `reg` (as an operand of `lane`) into `dst`.
            void Get(oaknut::WReg dst, const Decoded& d, Word reg, Word lane) {
                if (reg < 24) {
                    LoadField(dst, RegOffset(reg));
                    return;
                }
                switch (reg) {
                case 24: code.MOV(dst, oaknut::util::WZR); return;
                case 25: code.MOV(dst, Word(std::int32_t(std::int16_t(d.imm[lane])))); return;
                case 26: code.MOV(dst, Word(std::int32_t(std::int16_t(d.imm[1 - lane])))); return;
                case 27: code.MOV(dst, (d.imm[0] << 16) | d.imm[1]); return;
                case 28: code.MOV(dst, load0); return;
                case 29: code.MOV(dst, load1); return;
                default: code.B(fail_source); return;
                }
            }

            static bool IsFlow(Word op) {
                return op >= 23 && op <= 29;
            }

            // Where a loop can end (its back edge is checked only there) and which instructions
            // can run right after an END (only those check for the end of the macro).
            void Analyze(const std::vector<Decoded>& decoded) {
                const Word count = static_cast<Word>(decoded.size());
                loop_end.assign(count, false);
                after_end.assign(count, false);
                const auto mark = [&](std::vector<bool>& set, std::int64_t pc) {
                    if (pc >= 0 && pc < count) set[static_cast<std::size_t>(pc)] = true;
                };
                for (Word pc = 0; pc < count; ++pc) {
                    for (Word lane = 0; lane < 2; ++lane) {
                        const Decoded& d = decoded[pc];
                        if (d.op[lane] == 23 && d.imm[lane] >= 2 && pc + d.imm[lane] - 1 < count) {
                            loop_end[pc + d.imm[lane] - 1] = true;
                        }
                    }
                }
                for (Word pc = 0; pc < count; ++pc) {
                    const Decoded& d = decoded[pc];
                    if (!d.end) continue;
                    mark(after_end, std::int64_t{pc} + 1);
                    for (Word lane = 0; lane < 2; ++lane) {
                        const Word imm = d.imm[lane];
                        switch (d.op[lane]) {
                        case 23: mark(after_end, std::int64_t{pc} + imm); break;
                        case 24: mark(after_end, std::int64_t{pc} + (imm & 0x7fff)); break;
                        case 25: case 26: case 27: case 28: case 29:
                            mark(after_end, std::int64_t{pc} + (std::int32_t(imm << 19) >> 19));
                            break;
                        default: break;
                        }
                    }
                }
                // An END at a loop's last instruction can be followed by the loop's second one.
                for (Word pc = 0; pc < count; ++pc) {
                    for (Word lane = 0; lane < 2; ++lane) {
                        const Decoded& d = decoded[pc];
                        if (d.op[lane] == 23 && d.imm[lane] >= 2) {
                            const Word end = pc + d.imm[lane] - 1;
                            if (end < count && decoded[end].end) mark(after_end, std::int64_t{pc} + 1);
                        }
                    }
                }
            }

            void Instruction(Word pc, Word count, const Decoded& d) {
                // FIFO loads, in lane order.
                for (Word i = 0; i < 2; ++i) {
                    const oaknut::WReg load = i == 0 ? load0 : load1;
                    if (!d.load[i]) {
                        code.MOV(load, oaknut::util::WZR);
                        continue;
                    }
                    LoadField(value, offsetof(State, consumed));
                    LoadField(t0, offsetof(State, params_size));
                    code.CMP(value, t0);
                    BranchFar(Cond::HS, fail_underflow);
                    code.LDR(xt1, st, static_cast<std::uint32_t>(offsetof(State, params)));
                    code.LDR(load, xt1, value, oaknut::IndexExt::UXTW, 2);
                    code.ADD(value, value, 1);
                    StoreField(value, offsetof(State, consumed));
                }
                if (!d.pred_mode) {
                    Body(pc, count, d, 15);
                    return;
                }
                Label when_false;
                Get(value, d, d.pred_reg, 0);
                code.CBZ(value, when_false);
                Body(pc, count, d, d.mask_true);
                code.l(when_false);
                Body(pc, count, d, d.mask_false);
            }

            void Body(Word pc, Word count, const Decoded& d, Word mask) {
                // An inactive lane's result reads as 0 when an output selects it.
                for (Word lane = 0; lane < 2; ++lane) {
                    if (mask & (1U << lane)) continue;
                    bool selected = false;
                    for (Word out = 0; out < 2; ++out) {
                        if (!(mask & (4U << out))) continue;
                        selected |= d.out_method[out] == lane + 1 || d.out_emit[out] == lane + 1;
                    }
                    if (selected) {
                        code.MOV(lane == 0 ? alu0 : alu1, oaknut::util::WZR);
                    }
                }
                const bool flow = ((mask & 1) && IsFlow(d.op[0])) || ((mask & 2) && IsFlow(d.op[1]));
                if (flow) {
                    StoreFieldImm(pc + 1, offsetof(State, next_pc));
                    code.STRB(oaknut::util::WZR, st, static_cast<std::uint32_t>(offsetof(State, cancel_end)));
                }
                // Lane 1's operands are read before lane 0 commits its destination (or calls out).
                const bool lane0_active = (mask & 1) != 0;
                const bool early_operands =
                    (mask & 2) && lane0_active &&
                    (d.op[0] == 22 || (d.dst[0] < 24 && (d.dst[0] == d.src_a[1] || d.dst[0] == d.src_b[1])));
                if (early_operands) {
                    Get(value, d, d.src_a[1], 1);
                    StoreField(value, offsetof(State, scratch_x));
                    Get(value, d, d.src_b[1], 1);
                    StoreField(value, offsetof(State, scratch_y));
                }
                for (Word lane = 0; lane < 2; ++lane) {
                    if (!(mask & (1U << lane))) {
                        continue;
                    }
                    if (lane == 1 && early_operands) {
                        LoadField(x, offsetof(State, scratch_x));
                        LoadField(y, offsetof(State, scratch_y));
                    } else {
                        Get(x, d, d.src_a[lane], lane);
                        Get(y, d, d.src_b[lane], lane);
                    }
                    Lane(pc, d, lane);
                    // w10 holds the lane's value.
                    code.MOV(lane == 0 ? alu0 : alu1, value);
                    if (d.dst[lane] < 24) {
                        StoreField(value, RegOffset(d.dst[lane]));
                    } else if (d.dst[lane] != 24) {
                        code.B(fail_destination);
                    }
                }
                for (Word lane = 0; lane < 2; ++lane) {
                    if (!(mask & (4U << lane))) {
                        continue;
                    }
                    if (d.out_method[lane]) {
                        Output(value, d, d.out_method[lane]);
                        code.MOV(t1, 0xfff);
                        code.AND(t0, value, t1);
                        StoreField(t0, offsetof(State, method));
                        code.LSR(value, value, 12);
                        code.MOV(t1, 15);
                        code.AND(value, value, t1);
                        StoreField(value, offsetof(State, increment));
                    }
                    if (d.out_emit[lane]) {
                        Output(value, d, d.out_emit[lane]);
                        Label call_emit, emitted;
                        if (program.batch_emits) {
                            // Into the buffer unless it is full or the method is one the macro
                            // sees the effect of (DMA reads) or one `emit` may refuse.
                            LoadField(t0, offsetof(State, method));
                            LoadField(t1, offsetof(State, emit_count));
                            LoadField(t2, offsetof(State, emit_capacity));
                            code.CMP(t1, t2);
                            code.B(Cond::HS, call_emit);
                            code.MOV(t2, 0xe00);
                            code.CMP(t0, t2);
                            code.B(Cond::HS, call_emit);
                            code.SUB(t2, t0, 0x157);
                            code.CMP(t2, 1);
                            code.B(Cond::LS, call_emit);
                            code.ADD(xt3, st, static_cast<std::uint32_t>(offsetof(State, emit_buffer)));
                            code.ADD(xt3, xt3, xt1, oaknut::AddSubShift::LSL, 3);
                            code.STR(t0, xt3, 0);
                            code.STR(value, xt3, 4);
                            code.ADD(t1, t1, 1);
                            StoreField(t1, offsetof(State, emit_count));
                            code.B(emitted);
                            code.l(call_emit);
                        }
                        code.MOV(oaknut::util::W2, value);
                        LoadField(oaknut::util::W1, offsetof(State, method));
                        code.MOV(oaknut::util::X0, st);
                        code.LDR(call, st, static_cast<std::uint32_t>(offsetof(State, emit)));
                        code.BLR(call);
                        CbzFar(oaknut::util::W0, fail_emit);
                        if (program.batch_emits) {
                            code.l(emitted);
                        }
                        LoadField(value, offsetof(State, increment));
                        LoadField(t0, offsetof(State, method));
                        code.ADD(t0, t0, value);
                        StoreField(t0, offsetof(State, method));
                    }
                }
                // An END in the previous instruction ends the macro after this one.
                if (after_end[pc]) {
                    LoadByte(value, offsetof(State, end_next));
                    CbnzFar(value, success);
                }
                if (flow) {
                    LoadByte(value, offsetof(State, stopped));
                    CbnzFar(value, success);
                }
                if (d.end) {
                    if (flow) {
                        LoadByte(value, offsetof(State, cancel_end));
                        code.MOV(t0, 1);
                        code.EOR(value, value, t0);
                        StoreByte(value, offsetof(State, end_next));
                    } else {
                        StoreByteImm(1, offsetof(State, end_next));
                    }
                }
                if (loop_end[pc]) {
                    Label no_loop;
                    LoadField(value, offsetof(State, loop_count));
                    code.CBZ(value, no_loop);
                    LoadField(t0, offsetof(State, loop_end));
                    code.MOV(t1, pc);
                    code.CMP(t0, t1);
                    code.B(Cond::NE, no_loop);
                    code.SUB(value, value, 1);
                    StoreField(value, offsetof(State, loop_count));
                    LoadField(value, offsetof(State, loop_start));
                    code.ADD(value, value, 1);
                    code.B(dispatch);
                    code.l(no_loop);
                }
                if (flow) {
                    LoadField(value, offsetof(State, next_pc));
                    if (pc + 1 < count) {
                        code.MOV(t0, pc + 1);
                        code.CMP(value, t0);
                        BranchFar(Cond::NE, dispatch);
                        code.B(labels[pc + 1]);
                    } else {
                        code.B(dispatch);
                    }
                } else if (pc + 1 < count) {
                    code.B(labels[pc + 1]);
                } else {
                    code.B(fail_outside);
                }
            }

            void Output(oaknut::WReg dst, const Decoded& d, Word selector) {
                switch (selector) {
                case 1: code.MOV(dst, alu0); return;
                case 2: code.MOV(dst, alu1); return;
                case 3: code.MOV(dst, load0); return;
                case 4: code.MOV(dst, load1); return;
                case 5: code.MOV(dst, d.imm[0]); return;
                case 6: code.MOV(dst, d.imm[1]); return;
                case 8: code.MOV(dst, d.imm[0] >> 12); return;
                case 9: code.MOV(dst, d.imm[1] >> 12); return;
                case 10: code.MOV(dst, (d.imm[0] << 16) | d.imm[1]); return;
                default: code.B(fail_output); return;
                }
            }

            // w8 = x, w9 = y; leaves 1 or 0 in w10.
            void Compare(Word op) {
                code.CMP(x, y);
                switch (op) {
                case 17: case 25: code.CSET(value, Cond::LT); break;
                case 18: case 26: code.CSET(value, Cond::LO); break;
                case 19: case 27: code.CSET(value, Cond::LE); break;
                case 20: case 28: code.CSET(value, Cond::LS); break;
                default: code.CSET(value, Cond::EQ); break;
                }
            }

            void Lane(Word pc, const Decoded& d, Word lane) {
                const Word imm = d.imm[lane];
                const Word op = d.op[lane];
                switch (op) {
                case 0:
                    code.ADDS(value, x, y);
                    code.CSET(t0, Cond::CS);
                    StoreField(t0, offsetof(State, carry));
                    return;
                case 1:
                    LoadField(t0, offsetof(State, carry));
                    code.ADD(value, x, y);
                    code.ADD(value, value, t0);
                    return;
                case 2:
                    code.SUBS(value, x, y);
                    code.CSET(t0, Cond::CC);
                    StoreField(t0, offsetof(State, carry));
                    return;
                case 3:
                    LoadField(t0, offsetof(State, carry));
                    code.SUB(value, x, y);
                    code.SUB(value, value, t0);
                    return;
                case 4:
                    code.SMULL(xvalue, x, y);
                    code.LSR(xt0, xvalue, 32);
                    StoreField(t0, offsetof(State, carry));
                    return;
                case 6:
                    code.UMULL(xvalue, x, y);
                    code.LSR(xt0, xvalue, 32);
                    StoreField(t0, offsetof(State, carry));
                    return;
                case 5:
                    LoadField(value, offsetof(State, carry));
                    return;
                case 7:
                    code.MOV(t0, 0x1000);
                    code.CMP(x, t0);
                    BranchFar(Cond::NE, fail_extended);
                    code.MOV(value, oaknut::util::WZR);
                    return;
                case 8:
                    code.CLZ(value, x);
                    return;
                case 9:
                    code.LSLV(value, x, y);
                    return;
                case 10:
                    code.LSRV(value, x, y);
                    return;
                case 11:
                    code.ASRV(value, x, y);
                    return;
                case 12: code.AND(value, x, y); return;
                case 13: code.AND(value, x, y); code.MVN(value, value); return;
                case 14: code.ORR(value, x, y); return;
                case 15: code.EOR(value, x, y); return;
                case 16: {
                    const Word dest = (imm >> 10) & 63, bit_count = (imm >> 5) & 31;
                    const Word source = imm & 31;
                    if (dest >= 32) {
                        code.B(fail_merge);
                        return;
                    }
                    const Word bits_mask = (1U << bit_count) - 1;
                    code.MOV(t0, ~(bits_mask << dest));
                    code.AND(value, x, t0);
                    code.LSR(t0, y, source);
                    code.MOV(t1, bits_mask);
                    code.AND(t0, t0, t1);
                    code.LSL(t0, t0, dest);
                    code.ORR(value, value, t0);
                    return;
                }
                case 17: case 18: case 19: case 20: case 21:
                    Compare(op);
                    code.NEG(value, value);
                    return;
                case 22:
                    code.ADD(value, x, y);
                    code.MOV(t0, 0xffff);
                    code.AND(oaknut::util::W1, value, t0);
                    code.MOV(oaknut::util::X0, st);
                    code.LDR(call, st, static_cast<std::uint32_t>(offsetof(State, read)));
                    code.BLR(call);
                    code.MOV(value, oaknut::util::W0);
                    return;
                case 23: {
                    if (imm < 2) {
                        code.B(fail_loop);
                        return;
                    }
                    Label skip, done;
                    LoadField(t0, offsetof(State, loop_count));
                    CbnzFar(t0, fail_loop);
                    code.CBZ(x, skip);
                    code.SUB(t0, x, 1);
                    StoreField(t0, offsetof(State, loop_count));
                    StoreFieldImm(pc, offsetof(State, loop_start));
                    StoreFieldImm(pc + imm - 1, offsetof(State, loop_end));
                    code.B(done);
                    code.l(skip);
                    StoreFieldImm(pc + imm, offsetof(State, next_pc));
                    code.l(done);
                    code.MOV(value, oaknut::util::WZR);
                    return;
                }
                case 24:
                    StoreFieldImm(pc + (imm & 0x7fff), offsetof(State, next_pc));
                    code.MOV(value, oaknut::util::WZR);
                    return;
                case 25: case 26: case 27: case 28: case 29: {
                    Label not_taken;
                    Compare(op);
                    code.CMP(value, (imm & 0x8000) ? 1 : 0);
                    code.B(Cond::NE, not_taken);
                    const auto offset = std::int32_t(imm << 19) >> 19;
                    if (offset == -4096) {
                        StoreByteImm(1, offsetof(State, stopped));
                    } else {
                        StoreFieldImm(Word(std::int32_t(pc) + offset), offsetof(State, next_pc));
                        StoreByteImm((imm & 0x4000) ? 1 : 0, offsetof(State, cancel_end));
                    }
                    code.l(not_taken);
                    code.MOV(value, oaknut::util::WZR);
                    return;
                }
                case 30:
                    code.MOV(t0, 0xc00);
                    code.CMP(x, t0);
                    BranchFar(Cond::HS, fail_dread);
                    code.LDR(xt0, st, static_cast<std::uint32_t>(offsetof(State, dram)));
                    code.LDR(value, xt0, x, oaknut::IndexExt::UXTW, 2);
                    return;
                case 31:
                    code.MOV(t0, 0xc00);
                    code.CMP(x, t0);
                    BranchFar(Cond::HS, fail_dwrite);
                    code.LDR(xt0, st, static_cast<std::uint32_t>(offsetof(State, dram)));
                    code.STR(y, xt0, x, oaknut::IndexExt::UXTW, 2);
                    code.MOV(value, oaknut::util::WZR);
                    return;
                default:
                    code.MOV(value, oaknut::util::WZR);
                    return;
                }
            }
        };

        bool batch_emits{};
        oaknut::CodeBlock mem;
    };

    std::array<std::unique_ptr<Program>, 2> programs; ///< Plain, batching
    std::array<std::uint64_t, 2> compiled_generations{~std::uint64_t{0}, ~std::uint64_t{0}};
    Program* program{}; ///< The one the last Prepare chose
    bool compiled_batch{};
    std::unique_ptr<State> kept; ///< Options::keep_state
    bool kept_busy{};
};

} // namespace Tegra
