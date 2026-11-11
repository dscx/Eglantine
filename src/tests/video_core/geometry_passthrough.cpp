// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"

namespace {
using namespace Shader;

IR::Program PassthroughProgram() {
    IR::Program program;
    program.stage = Stage::Geometry;
    program.is_geometry_passthrough = true;
    program.invocations = 1;
    // The hardware passes position and generic 0 through; the lowering stores them.
    for (u32 component = 0; component < 4; ++component) {
        program.info.passthrough.Set(IR::Attribute::PositionX + component);
        program.info.passthrough.Set(IR::Attribute::Generic0X + component);
        program.info.stores.Set(IR::Attribute::PositionX + component);
        program.info.stores.Set(IR::Attribute::Generic0X + component);
    }
    return program;
}
} // Anonymous namespace

TEST_CASE("Geometry passthrough: a program that only forwards attributes is pure", "[geometry]") {
    REQUIRE(Maxwell::IsPureGeometryPassthrough(PassthroughProgram()));
}

TEST_CASE("Geometry passthrough: a body that stores its own attribute is not", "[geometry]") {
    IR::Program program{PassthroughProgram()};
    program.info.stores.Set(IR::Attribute::Layer);
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(program));
    IR::Program viewport{PassthroughProgram()};
    viewport.info.stores.Set(IR::Attribute::ViewportIndex);
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(viewport));
}

TEST_CASE("Geometry passthrough: memory writes and instancing are not", "[geometry]") {
    IR::Program global{PassthroughProgram()};
    global.info.stores_global_memory = true;
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(global));
    IR::Program ssbo{PassthroughProgram()};
    ssbo.info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0, .count = 1, .is_written = true});
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(ssbo));
    IR::Program read_only{PassthroughProgram()};
    read_only.info.storage_buffers_descriptors.push_back({.cbuf_index = 0, .cbuf_offset = 0, .count = 1, .is_written = false});
    REQUIRE(Maxwell::IsPureGeometryPassthrough(read_only));
    IR::Program instanced{PassthroughProgram()};
    instanced.invocations = 2;
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(instanced));
    IR::Program ordinary{PassthroughProgram()};
    ordinary.is_geometry_passthrough = false;
    REQUIRE(!Maxwell::IsPureGeometryPassthrough(ordinary));
}
