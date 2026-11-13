// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/spirv_blocks.h"

namespace {

constexpr u32 Word(u32 length, u32 opcode) {
    return (length << 16) | opcode;
}

constexpr u32 OpFunctionEnd = 56;
constexpr u32 OpLabel = 248;
constexpr u32 OpBranch = 249;
constexpr u32 OpReturn = 253;

std::vector<u32> Module(std::initializer_list<u32> body) {
    // Magic, version, generator, id bound, reserved.
    std::vector<u32> words{0x07230203, 0x00010500, 0, 100, 0};
    words.insert(words.end(), body);
    return words;
}

} // Anonymous namespace

// The counter walks the instruction stream by each instruction's length word, so a label's
// operand (its result id) is never mistaken for an opcode.
TEST_CASE("SPIR-V basic blocks are counted from OpLabel instructions",
          "[video_core][vulkan][spirv_blocks]") {
    const auto module = Module({Word(2, OpLabel), 7, Word(2, OpBranch), 9, Word(2, OpLabel), 9,
                                Word(1, OpReturn), Word(1, OpFunctionEnd)});
    CHECK(Vulkan::CountSpirvBlocks(module) == 2);

    CHECK(Vulkan::CountSpirvBlocks(Module({})) == 0);
    CHECK(Vulkan::CountSpirvBlocks(std::array<u32, 3>{1, 2, 3}) == 0);

    // A label id that happens to equal the OpLabel opcode is an operand, not a block.
    const auto operand = Module({Word(2, OpLabel), OpLabel, Word(1, OpReturn)});
    CHECK(Vulkan::CountSpirvBlocks(operand) == 1);

    // A zero length word would loop forever; the scan stops there.
    const auto malformed =
        Module({Word(2, OpLabel), 7, Word(0, OpLabel), Word(2, OpLabel), 8});
    CHECK(Vulkan::CountSpirvBlocks(malformed) == 1);
}
