// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/vulkan_common/vulkan_device.h"

// Mesa RADV hangs the GPU on its mesh pipelines (issues #266, #341, #404), so VK_EXT_mesh_shader
// stays off there unless the user opts in. Every other driver keeps the extension when it has it.
TEST_CASE("Mesh shaders stay off by default on RADV", "[vulkan][mesh]") {
    CHECK_FALSE(Vulkan::MeshShadersDefaultOn(VK_DRIVER_ID_MESA_RADV, false));
    CHECK(Vulkan::MeshShadersDefaultOn(VK_DRIVER_ID_MESA_RADV, true));
    for (const VkDriverId id : {VK_DRIVER_ID_AMD_PROPRIETARY, VK_DRIVER_ID_NVIDIA_PROPRIETARY,
                                VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA, VK_DRIVER_ID_MOLTENVK,
                                VK_DRIVER_ID_MESA_TURNIP}) {
        CHECK(Vulkan::MeshShadersDefaultOn(id, false));
    }
}
