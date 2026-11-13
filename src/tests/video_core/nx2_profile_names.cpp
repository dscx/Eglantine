// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <set>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_profile.h"

namespace {

template <typename Names>
void CheckNames(const Names& names) {
    std::set<std::string_view> seen;
    for (const char* name : names) {
        REQUIRE(name != nullptr);
        CHECK_FALSE(std::string_view{name}.empty());
        CHECK(seen.insert(name).second);
    }
}

} // Anonymous namespace

// The name tables are sized by their enums' Count, so a zone added without a name is a null
// pointer the report would print; every entry must be present and distinct.
TEST_CASE("Every profile zone, flush origin and drain site has its own name",
          "[video_core][profile]") {
    CheckNames(VideoCommon::Nx2Profile::zone_names);
    CheckNames(VideoCommon::Nx2Profile::flush_origin_names);
    CheckNames(VideoCommon::drain_site_names);
    STATIC_REQUIRE(static_cast<size_t>(VideoCommon::Nx2Profile::Counter::Count) <=
                   std::tuple_size_v<decltype(VideoCommon::Nx2Profile::ThreadData::counters)>);
}
