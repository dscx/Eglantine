// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "video_core/engines/maxwell_3d.h"

using Regs = Tegra::Engines::Maxwell3D::Regs;

TEST_CASE("Maxwell draw state: depth and stencil writes", "[video_core]") {
    auto regs = std::make_unique<Regs>();
    std::memset(regs.get(), 0, sizeof(Regs));
    REQUIRE_FALSE(regs->WritesDepthStencil());

    SECTION("a depth write needs the depth test") {
        regs->depth_write_enabled = 1;
        REQUIRE_FALSE(regs->WritesDepthStencil());
        regs->depth_test_enable = 1;
        REQUIRE(regs->WritesDepthStencil());
        regs->depth_write_enabled = 0;
        REQUIRE_FALSE(regs->WritesDepthStencil());
    }
    SECTION("a stencil write needs a mask") {
        regs->stencil_enable = 1;
        REQUIRE_FALSE(regs->WritesDepthStencil());
        regs->stencil_front_mask = 0xff;
        REQUIRE(regs->WritesDepthStencil());
    }
    SECTION("the back mask counts only when two-sided") {
        regs->stencil_enable = 1;
        regs->stencil_back_mask = 0xff;
        REQUIRE_FALSE(regs->WritesDepthStencil());
        regs->stencil_two_side_enable = 1;
        REQUIRE(regs->WritesDepthStencil());
    }
}
