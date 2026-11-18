// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/deferred_submit_state.h"

namespace {

using namespace VideoCommon::Nx2Perf;
using Vulkan::DeferredSubmitState;
using Due = DeferredSubmitState::Due;

constexpr size_t Index(Opt opt) {
    return static_cast<size_t>(opt);
}

} // namespace

TEST_CASE("Deferred waited submit: pass boundary, draw bound or dropped", "[video_core]") {
    DeferredSubmitState state;
    REQUIRE(!state.Pending());
    REQUIRE(state.Check(70, true, 32) == Due::No);

    SECTION("the next draw outside a pass submits") {
        state.Request(64);
        REQUIRE(state.Pending());
        REQUIRE(state.Check(70, true, 32) == Due::No);
        REQUIRE(state.Pending());
        REQUIRE(state.Check(80, false, 32) == Due::Boundary);
        REQUIRE(!state.Pending());
    }
    SECTION("the bound submits inside the pass") {
        state.Request(64);
        REQUIRE(state.Check(95, true, 32) == Due::No);
        REQUIRE(state.Check(96, true, 32) == Due::Bound);
        REQUIRE(!state.Pending());
    }
    SECTION("a submission by anything else drops the request") {
        state.Request(64);
        REQUIRE(state.Check(3, true, 32) == Due::Dropped);
        REQUIRE(!state.Pending());
    }
    SECTION("a second request keeps the first count") {
        state.Request(64);
        state.Request(90);
        REQUIRE(state.Check(95, true, 32) == Due::No);
        REQUIRE(state.Check(96, true, 32) == Due::Bound);
    }
}

TEST_CASE("Nx2Perf waited_submit_defer is on by default and off under SAFE", "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::WaitedSubmitDefer)] == "waited_submit_defer");
    REQUIRE(!IsOff(StartOffMask(false, "", ""), Index(Opt::WaitedSubmitDefer)));
    REQUIRE(IsOff(StartOffMask(true, "", ""), Index(Opt::WaitedSubmitDefer)));
    REQUIRE(!IsOff(StartOffMask(true, "waited_submit_defer", ""), Index(Opt::WaitedSubmitDefer)));
    REQUIRE(IsOff(StartOffMask(false, "", "waited_submit_defer"), Index(Opt::WaitedSubmitDefer)));
}
