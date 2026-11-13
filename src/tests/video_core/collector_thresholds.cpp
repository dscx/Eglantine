// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "common/literals.h"
#include "video_core/texture_cache/texture_cache_base.h"

using namespace Common::Literals;
using VideoCommon::CollectorThresholdCaps;

namespace {

constexpr u64 Floor = 5_GiB;
constexpr u64 NoCap = std::numeric_limits<u64>::max();

} // Anonymous namespace

// Without a budget, or with one of 2 GiB or less, only the floor caps the collector.
TEST_CASE("Small or unknown budgets leave the floor alone",
          "[video_core][texture_cache][collector]") {
    auto caps = CollectorThresholdCaps(0, Floor);
    CHECK(caps.critical == Floor);
    CHECK(caps.expected == NoCap);

    caps = CollectorThresholdCaps(2_GiB, Floor);
    CHECK(caps.critical == Floor);
    CHECK(caps.expected == NoCap);
}

// Past 2 GiB and below the floor, critical is 3/4 and expected 3/5 of the budget (PR 250's clamp:
// the driver pages textures out well before the budget is full).
TEST_CASE("Budgets under the floor cap the collector at 3/4 and 3/5",
          "[video_core][texture_cache][collector]") {
    auto caps = CollectorThresholdCaps(3686_MiB, Floor);
    CHECK(caps.critical == 2898788352ULL);
    CHECK(caps.expected == 2319030681ULL);

    caps = CollectorThresholdCaps(5632_MiB, Floor);
    CHECK(caps.critical == 4224_MiB);
    CHECK(caps.expected == 3543348018ULL);

    caps = CollectorThresholdCaps(6144_MiB, Floor);
    CHECK(caps.critical == 4608_MiB);
    CHECK(caps.expected == 3865470564ULL);
}

// From about 6.7 GiB the floor, then 5/8 of the budget, are the tighter critical cap; expected
// keeps the 3/5 cap, which the branch's 768 MiB gap under critical already meets on an 8 GB card.
TEST_CASE("Larger budgets keep the floor and the 5/8 rule",
          "[video_core][texture_cache][collector]") {
    auto caps = CollectorThresholdCaps(7383_MiB, Floor);
    CHECK(caps.critical == Floor);
    CHECK(caps.expected == 4644981963ULL);

    caps = CollectorThresholdCaps(11469_MiB, Floor);
    CHECK(caps.critical == 7516323840ULL);
    CHECK(caps.expected == 7215670884ULL);

    caps = CollectorThresholdCaps(22528_MiB, Floor);
    CHECK(caps.critical == 14080_MiB);
    CHECK(caps.expected == 14173392075ULL);
}
