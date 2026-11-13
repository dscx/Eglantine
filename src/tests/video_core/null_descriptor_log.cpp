// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "video_core/textures/texture.h"

using Tegra::Texture::TICEntry;

namespace {

TICEntry Zeroed() {
    TICEntry entry{};
    entry.raw.fill(0);
    return entry;
}

} // Anonymous namespace

// A format-0 TIC is a null descriptor. The zeroed one titles leave in unused slots is the
// placeholder; one with an address or an extent describes a texture whose format is unknown.
TEST_CASE("The zeroed descriptor is the placeholder, an addressed or sized one is not",
          "[video_core][texture][null_descriptor]") {
    CHECK(Tegra::Texture::IsPlaceholderDescriptor(Zeroed()));

    TICEntry block_linear = Zeroed();
    block_linear.header_version.Assign(Tegra::Texture::TICHeaderVersion::BlockLinear);
    CHECK(Tegra::Texture::IsPlaceholderDescriptor(block_linear));

    TICEntry addressed = Zeroed();
    addressed.address_low = 0x1000;
    CHECK_FALSE(Tegra::Texture::IsPlaceholderDescriptor(addressed));

    TICEntry sized = Zeroed();
    sized.header_version.Assign(Tegra::Texture::TICHeaderVersion::BlockLinear);
    sized.width_minus_one.Assign(1023);
    sized.height_minus_1.Assign(1023);
    CHECK_FALSE(Tegra::Texture::IsPlaceholderDescriptor(sized));
}

TEST_CASE("Each distinct descriptor is noted once, up to the bound",
          "[video_core][texture][null_descriptor]") {
    Tegra::Texture::NullDescriptorLog log;
    const TICEntry first = Zeroed();
    CHECK(log.Note(first));
    CHECK_FALSE(log.Note(first));

    TICEntry second = Zeroed();
    second.raw[3] = 1;
    CHECK(log.Note(second));
    CHECK_FALSE(log.Note(second));
    CHECK_FALSE(log.Note(first));

    for (u64 n = 2; n < Tegra::Texture::NullDescriptorLog::MaxEntries; ++n) {
        TICEntry entry = Zeroed();
        entry.raw[3] = n;
        CHECK(log.Note(entry));
    }
    TICEntry extra = Zeroed();
    extra.raw[3] = 0xffff;
    CHECK_FALSE(log.Note(extra));
    CHECK_FALSE(log.Note(first));
}

TEST_CASE("The description carries the raw words and the address",
          "[video_core][texture][null_descriptor]") {
    TICEntry entry = Zeroed();
    entry.address_low = 0x2000;
    entry.address_high.Assign(0x1);
    const std::string text = Tegra::Texture::DescribeDescriptor(entry);
    CHECK(text.find("addr 0x100002000") != std::string::npos);
    CHECK(text.find("raw ") != std::string::npos);
    CHECK(text.find("1x1x1") != std::string::npos);
}
