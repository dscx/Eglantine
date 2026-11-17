// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// A guest wait for idle met inside a render pass. Its full barrier cannot be recorded in the
/// pass, so the scheduler records it when the pass ends; the deferred fragment barrier ends the
/// pass early for a draw that reads what the pass stored.
class WaitForIdleState {
public:
    /// Remembers a wait whose barrier covers `stages`; returns how many waits are pending, so a
    /// run of them inside one pass costs one barrier.
    u32 Defer(u32 stages) {
        pending_stages |= stages;
        return ++deferred;
    }

    bool Pending() const {
        return deferred != 0;
    }

    /// The stages of the pending barrier, and clears the state.
    u32 Take() {
        const u32 stages = pending_stages;
        pending_stages = 0;
        deferred = 0;
        return stages;
    }

private:
    u32 pending_stages{};
    u32 deferred{};
};

} // namespace Vulkan
