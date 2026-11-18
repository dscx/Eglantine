// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/wait_for_idle_state.h"

namespace {

using namespace VideoCommon::Nx2Perf;

constexpr size_t Index(Opt opt) {
    return static_cast<size_t>(opt);
}

} // namespace

TEST_CASE("Deferred wait for idle: one barrier for a run of waits", "[video_core]") {
    Vulkan::WaitForIdleState state;
    REQUIRE(!state.Pending());
    REQUIRE(state.Take() == 0);

    SECTION("a deferred wait is pending until taken, with its stages") {
        REQUIRE(state.Defer(0x1) == 1);
        REQUIRE(state.Pending());
        REQUIRE(state.Take() == 0x1);
        REQUIRE(!state.Pending());
        REQUIRE(state.Take() == 0);
    }

    SECTION("waits in the same pass merge their stages into one barrier") {
        REQUIRE(state.Defer(0x1) == 1);
        REQUIRE(state.Defer(0x4) == 2);
        REQUIRE(state.Defer(0x4) == 3);
        REQUIRE(state.Pending());
        REQUIRE(state.Take() == 0x5);
        REQUIRE(!state.Pending());
    }
}

TEST_CASE("Nx2Perf wfi_defer is off until EDEN_NX2_PERF_ON names it", "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::WfiDefer)] == "wfi_defer");
    REQUIRE(IsOff(OffUntilNamedMask(), Index(Opt::WfiDefer)));
    // ParseOffMask alone keeps its contract: normal mode turns nothing off.
    REQUIRE(ParseOffMask(false, "", "") == u64{0});
    REQUIRE(StartOffMask(false, "", "") == OffUntilNamedMask());
    REQUIRE(IsOff(StartOffMask(true, "", ""), Index(Opt::WfiDefer)));
    REQUIRE(!IsOff(StartOffMask(false, "", ""), Index(Opt::FeedbackDepthReadOnly)));
    REQUIRE(!IsOff(StartOffMask(false, "wfi_defer", ""), Index(Opt::WfiDefer)));
    REQUIRE(!IsOff(StartOffMask(true, "wfi_defer", ""), Index(Opt::WfiDefer)));
    REQUIRE(IsOff(StartOffMask(false, "wfi_defer", "wfi_defer"), Index(Opt::WfiDefer)));
    REQUIRE(StartOffMask(false, "", "bind_memo") == (OffUntilNamedMask() | Bit(Opt::BindMemo)));
}
