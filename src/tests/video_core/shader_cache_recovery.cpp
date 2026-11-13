// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "shader_recompiler/stage.h"
#include "video_core/shader_environment.h"

namespace {

namespace fs = std::filesystem;

constexpr u32 CacheVersion = 35;
constexpr u32 MinCompatibleVersion = 20;
constexpr std::array<char, 8> Magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

// An empty directory, removed with everything in it at the end of the test.
struct ScratchDir {
    fs::path path = fs::temp_directory_path() /
                    ("shader-cache-recovery-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ScratchDir() {
        fs::create_directories(path);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

template <typename T>
void Put(std::string& out, const T& value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string Header(u32 version) {
    std::string out;
    out.append(Magic.data(), Magic.size());
    Put(out, version);
    return out;
}

// One compute pipeline record as GenericEnvironment::Serialize writes it: a single environment
// claiming `claimed_code_size` bytes of code with `code_bytes` of them present, no cached answers,
// and the pipeline key (eight bytes, read by the load callback).
std::string ComputeRecord(u64 claimed_code_size, size_t code_bytes, u64 key) {
    std::string out;
    Put(out, u32{1}); // environments in the record
    Put(out, claimed_code_size);
    Put(out, u64{0}); // texture types
    Put(out, u64{0}); // texture pixel formats
    Put(out, u64{0}); // constant buffer values
    Put(out, u64{0}); // storage pointer types
    Put(out, u64{0}); // constant buffer replacements
    Put(out, u32{0}); // local memory size
    Put(out, u32{0}); // texture bound buffer
    Put(out, u32{0}); // start address
    Put(out, u32{0}); // read lowest
    Put(out, u32{0}); // read highest
    Put(out, u32{1}); // viewport transform state
    Put(out, static_cast<u32>(Shader::Stage::Compute));
    out.append(code_bytes, '\0');
    Put(out, std::array<u32, 3>{1, 1, 1}); // workgroup size
    Put(out, u32{0});                      // shared memory size
    Put(out, key);
    return out;
}

std::string ComputeRecord(u64 key) {
    return ComputeRecord(16, 16, key);
}

void WriteFile(const fs::path& path, const std::string& content) {
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::vector<u64> Load(const fs::path& file, VideoCommon::CacheUpgrade* upgrade = nullptr) {
    std::vector<u64> keys;
    std::stop_source stop;
    VideoCommon::LoadPipelines(
        stop.get_token(), file, CacheVersion,
        [&keys](std::ifstream& stream, VideoCommon::FileEnvironment) {
            u64 key{};
            stream.read(reinterpret_cast<char*>(&key), sizeof(key));
            keys.push_back(key);
        },
        [](std::ifstream&, std::vector<VideoCommon::FileEnvironment>) {
            FAIL("a compute-only cache produced a graphics record");
        },
        false, MinCompatibleVersion, upgrade);
    return keys;
}

} // Anonymous namespace

TEST_CASE("Pipeline cache loading keeps the records before a truncated one",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    const std::string valid = Header(CacheVersion) + ComputeRecord(1) + ComputeRecord(2);
    const std::string third = ComputeRecord(3);
    WriteFile(file, valid + third.substr(0, third.size() / 2));

    CHECK(Load(file) == std::vector<u64>{1, 2});
    CHECK(fs::file_size(file) == valid.size());
    // The cut-back file loads again in full.
    CHECK(Load(file) == std::vector<u64>{1, 2});
    CHECK(fs::file_size(file) == valid.size());
}

TEST_CASE("Pipeline cache loading rejects a record claiming more code than the file holds",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    const std::string valid = Header(CacheVersion) + ComputeRecord(1);
    WriteFile(file, valid + ComputeRecord(u64{1} << 40, 0, 2));

    CHECK(Load(file) == std::vector<u64>{1});
    CHECK(fs::file_size(file) == valid.size());
}

TEST_CASE("Pipeline cache loading stops at a record without environments",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    const std::string valid = Header(CacheVersion) + ComputeRecord(1);
    std::string empty_record;
    Put(empty_record, u32{0});
    WriteFile(file, valid + empty_record + ComputeRecord(2));

    CHECK(Load(file) == std::vector<u64>{1});
    CHECK(fs::file_size(file) == valid.size());
}

TEST_CASE("Pipeline cache with an unreadable header is moved aside, not deleted",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    const std::string garbage = "not a pipeline cache file at all";
    WriteFile(file, garbage);

    CHECK(Load(file).empty());
    CHECK_FALSE(fs::exists(file));
    auto quarantined = file;
    quarantined += ".invalid";
    REQUIRE(fs::exists(quarantined));
    CHECK(fs::file_size(quarantined) == garbage.size());
}

TEST_CASE("Pipeline cache shorter than its header is moved aside",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    WriteFile(file, "yuzu");

    CHECK(Load(file).empty());
    CHECK_FALSE(fs::exists(file));
    auto quarantined = file;
    quarantined += ".invalid";
    CHECK(fs::exists(quarantined));
}

TEST_CASE("Empty pipeline cache file is left in place", "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    WriteFile(file, "");

    CHECK(Load(file).empty());
    REQUIRE(fs::exists(file));
    CHECK(fs::file_size(file) == 0);
    // Nothing was moved aside either.
    CHECK(std::distance(fs::directory_iterator{dir.path}, fs::directory_iterator{}) == 1);
}

TEST_CASE("Pipeline cache older than the compatible versions is renamed to its version",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    WriteFile(file, Header(5) + ComputeRecord(1));

    CHECK(Load(file).empty());
    CHECK_FALSE(fs::exists(file));
    CHECK(fs::exists(dir.path / "vulkan.bin.v5"));
}

TEST_CASE("Compatible older pipeline cache is read for its upgrade up to a damaged tail",
          "[video_core][shader_cache_recovery]") {
    ScratchDir dir;
    const auto file = dir.path / "vulkan.bin";
    const std::string first = ComputeRecord(1);
    const std::string second = ComputeRecord(2);
    const std::string third = ComputeRecord(3);
    const std::string content = Header(30) + first + second + third.substr(0, third.size() / 2);
    WriteFile(file, content);

    VideoCommon::CacheUpgrade upgrade;
    CHECK(Load(file, &upgrade) == std::vector<u64>{1, 2});
    CHECK(upgrade.from_version == 30);
    REQUIRE(upgrade.records.size() == 2);
    CHECK(upgrade.records[0] == first);
    CHECK(upgrade.records[1] == second);
    CHECK(upgrade.is_compute == std::vector<bool>{true, true});
    // The file is left for the rewrite under the current version that follows.
    CHECK(fs::file_size(file) == content.size());
}
