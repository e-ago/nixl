// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <thread>
#include <cstdio>

int
main() {
    docaXferCompletion *slots;
    docaNotif *notifications;
    uint32_t *flags;
    docaXferReqGpu *request;
    if (cudaHostAlloc(&slots, 128 * sizeof(*slots), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc(&notifications, 3 * sizeof(*notifications), cudaHostAllocMapped) !=
            cudaSuccess ||
        cudaHostAlloc(&flags, 3 * sizeof(*flags), cudaHostAllocMapped) != cudaSuccess ||
        cudaMalloc(&request, sizeof(*request)) != cudaSuccess) {
        return 2;
    }
    memset(slots, 0, 128 * sizeof(*slots));
    memset(notifications, 0, 3 * sizeof(*notifications));
    memset(flags, 0, 3 * sizeof(*flags));
    cudaMemset(request, 0, sizeof(*request));
    cudaDeviceSynchronize();
    slots[0].xferReqRingGpu = request;
    slots[0].completed = 1; // Retired slot from the previous lap.
    cudaMemcpyToSymbol(test_gate, &flags, sizeof(flags));
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    kernel_progress<<<3, 1, 0, stream>>>(
        slots, notifications, notifications + 1, notifications + 2, flags + 2);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (*(volatile uint32_t *)flags == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            return 3;
        }
        std::this_thread::yield();
    }
    // The exact reset order used by postXfer, before its asynchronous launch.
    slots[0].xferReqRingGpu = nullptr;
    slots[0].completed = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    *(volatile uint32_t *)(flags + 1) = 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    *(volatile uint32_t *)(flags + 2) = 1;
    cudaError_t result = cudaStreamSynchronize(stream);
    printf("progress kernel result=%d %s\n", (int)result, cudaGetErrorString(result));
    // Process exit destroys this test's context even after a device fault.
    return result == cudaSuccess ? 0 : 1;
}
