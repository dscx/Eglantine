// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "dynarmic/interface/exclusive_monitor.h"

TEST_CASE("Exclusive monitor: a local reservation stands or falls on the compare-and-swap",
          "[core]") {
    Dynarmic::ExclusiveMonitor monitor{2};
    u64 memory = 5;
    REQUIRE(monitor.ReadAndMarkLocal<u64>(0, 0x1000, [&] { return memory; }) == 5);

    SECTION("the store sees the value loaded then and the reservation is consumed") {
        u64 seen = 0;
        REQUIRE(monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [&](u64 expected) {
            seen = expected;
            memory = 6;
            return true;
        }));
        REQUIRE(seen == 5);
        REQUIRE(memory == 6);
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [](u64) { return true; }));
    }
    SECTION("another address or another core has no reservation") {
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(0, 0x2000, [](u64) { return true; }));
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(1, 0x1000, [](u64) { return true; }));
    }
    SECTION("a failed compare-and-swap fails the store and still consumes the reservation") {
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [](u64) { return false; }));
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [](u64) { return true; }));
    }
    SECTION("another core's store does not clear this core's reservation") {
        REQUIRE(monitor.ReadAndMarkLocal<u64>(1, 0x1000, [&] { return memory; }) == 5);
        REQUIRE(monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [](u64) { return true; }));
        REQUIRE(monitor.DoExclusiveOperationLocal<u64>(1, 0x1000, [](u64) { return true; }));
    }
    SECTION("ClearProcessor drops the reservation") {
        monitor.ClearProcessor(0);
        REQUIRE(!monitor.DoExclusiveOperationLocal<u64>(0, 0x1000, [](u64) { return true; }));
    }
}
