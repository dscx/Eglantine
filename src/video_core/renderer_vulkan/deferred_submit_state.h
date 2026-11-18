// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Vulkan {

/// Nx2Perf waited_submit_defer: a submission the back wanted while the front waited (or the host
/// GPU had caught up), met inside a render pass. On Apple GPUs the submission would end the pass,
/// a tile store and reload. The pass stays open and the submission moves to the next draw that
/// begins outside a pass, or to a draw boundary `bound` draws after the request. A submission made
/// meanwhile by anything else (draws since submit below the request's) drops the request.
class DeferredSubmitState {
public:
    enum class Due { No, Boundary, Bound, Dropped };

    /// A submission wanted at `draws_since_submit`; a request already pending keeps its count.
    void Request(u32 draws_since_submit) {
        if (!pending) {
            pending = true;
            requested_at = draws_since_submit;
        }
    }

    [[nodiscard]] bool Pending() const noexcept {
        return pending;
    }

    /// At a draw boundary: whether the submission is due now and why. Clears the request when it
    /// is due or dropped.
    [[nodiscard]] Due Check(u32 draws_since_submit, bool pass_active, u32 bound) {
        if (!pending) {
            return Due::No;
        }
        if (draws_since_submit < requested_at) {
            pending = false;
            return Due::Dropped;
        }
        if (!pass_active) {
            pending = false;
            return Due::Boundary;
        }
        if (draws_since_submit - requested_at >= bound) {
            pending = false;
            return Due::Bound;
        }
        return Due::No;
    }

private:
    bool pending{};
    u32 requested_at{};
};

} // namespace Vulkan
