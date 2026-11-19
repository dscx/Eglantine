// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <boost/container/static_vector.hpp>

#include <dynarmic/common/spin_lock.h>

namespace Dynarmic {

using VAddr = std::uint64_t;
using Vector = std::array<std::uint64_t, 2>;

class ExclusiveMonitor {
public:
    /// @param processor_count Maximum number of processors using this global
    ///                        exclusive monitor. Each processor must have a
    ///                        unique id.
    explicit ExclusiveMonitor(size_t processor_count);

    size_t GetProcessorCount() const;

    /// Marks a region containing [address, address+size) to be exclusive to
    /// processor processor_id.
    template<typename T, typename Function>
    T ReadAndMark(size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        const VAddr masked_address = address & RESERVATION_GRANULE_MASK;

        Lock();
        exclusive_addresses[processor_id] = masked_address;
        const T value = op();
        std::memcpy(exclusive_values[processor_id].data(), &value, sizeof(T));
        Unlock();
        return value;
    }

    /// Checks to see if processor processor_id has exclusive access to the
    /// specified region. If it does, executes the operation then clears
    /// the exclusive state for processors if their exclusive region(s)
    /// contain [address, address+size).
    template<typename T, typename Function>
        requires std::is_trivially_copyable_v<T>
    bool DoExclusiveOperation(size_t processor_id, VAddr address, Function op) {
        if (!CheckAndClear(processor_id, address)) {
            return false;
        }

        T saved_value;
        std::memcpy(&saved_value, exclusive_values[processor_id].data(), sizeof(T));
        const bool result = op(saved_value);

        Unlock();
        return result;
    }

    /// Unsafe_IgnoreGlobalMonitor on the callback backends (arm64), the behaviour the x64 inline
    /// path has under the flag: the reservation is this processor's alone, taken without the
    /// lock and never clearing the other processors' reservations. The guest load comes before
    /// the reservation is published; the exclusive store then stands or falls on the
    /// compare-and-swap in its `op`. Only the owning processor's thread touches its slot here.
    template<typename T, typename Function>
    T ReadAndMarkLocal(size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        const T value = op();
        std::memcpy(exclusive_values[processor_id].data(), &value, sizeof(T));
        exclusive_addresses[processor_id] = address & RESERVATION_GRANULE_MASK;
        return value;
    }

    /// The store half of ReadAndMarkLocal: with the processor's reservation on `address`, runs
    /// `op` with the value loaded then and consumes the reservation; without one, fails.
    template<typename T, typename Function>
        requires std::is_trivially_copyable_v<T>
    bool DoExclusiveOperationLocal(size_t processor_id, VAddr address, Function op) {
        VAddr& reservation = exclusive_addresses[processor_id];
        if (reservation != (address & RESERVATION_GRANULE_MASK)) {
            return false;
        }
        reservation = INVALID_EXCLUSIVE_ADDRESS;
        T saved_value;
        std::memcpy(&saved_value, exclusive_values[processor_id].data(), sizeof(T));
        return op(saved_value);
    }

    /// Unmark everything.
    void Clear();
    /// Unmark processor id
    void ClearProcessor(size_t processor_id);

private:
    bool CheckAndClear(size_t processor_id, VAddr address);

    void Lock();
    void Unlock();

    friend volatile int* GetExclusiveMonitorLockPointer(ExclusiveMonitor*);
    friend size_t GetExclusiveMonitorProcessorCount(ExclusiveMonitor*);
    friend VAddr* GetExclusiveMonitorAddressPointer(ExclusiveMonitor*, size_t index);
    friend Vector* GetExclusiveMonitorValuePointer(ExclusiveMonitor*, size_t index);

    static constexpr VAddr RESERVATION_GRANULE_MASK = 0xFFFF'FFFF'FFFF'FFFFull;
    static constexpr VAddr INVALID_EXCLUSIVE_ADDRESS = 0xDEAD'DEAD'DEAD'DEADull;
    static constexpr size_t MAX_NUM_CPU_CORES = 6; // Sync with src/core/hardware_properties
    boost::container::static_vector<VAddr, MAX_NUM_CPU_CORES> exclusive_addresses;
    boost::container::static_vector<Vector, MAX_NUM_CPU_CORES> exclusive_values;
    SpinLock lock;
};

}  // namespace Dynarmic
