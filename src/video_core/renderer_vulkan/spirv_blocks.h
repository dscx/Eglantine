// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <span>

#include "common/common_types.h"

namespace Vulkan {

/// Basic blocks (OpLabel instructions) in a SPIR-V module. A zero length word ends the scan.
[[nodiscard]] constexpr size_t CountSpirvBlocks(std::span<const u32> code) {
    constexpr u32 OpLabel = 248;
    constexpr size_t HeaderWords = 5;
    size_t blocks = 0;
    for (size_t i = HeaderWords; i < code.size();) {
        const u32 word = code[i];
        const u32 length = word >> 16;
        if (length == 0) {
            break;
        }
        if ((word & 0xffff) == OpLabel) {
            ++blocks;
        }
        i += length;
    }
    return blocks;
}

/// MoltenVK converts SPIR-V with SPIRV-Cross, whose control-flow analysis recurses once per block
/// on the thread that creates the pipeline. A std::thread's 512 KiB stack on macOS died on an
/// 11,487-block compute program from Ocarina of Time's shader archives, while every program the
/// game draws with has under 700. A stage past this many blocks is refused there instead of
/// taking the process down. EDEN_NX2_MVK_MAX_BLOCKS overrides it; 0 disables the check.
[[nodiscard]] inline size_t MoltenVkBlockBudget() {
    static const size_t budget = [] {
        const char* const value = std::getenv("EDEN_NX2_MVK_MAX_BLOCKS");
        return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
                                : size_t{2048};
    }();
    return budget;
}

} // namespace Vulkan
