#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a controlled interleaving test from the real GPUNETIO progress kernel."""

import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, help="Kernel source to test; defaults to this checkout"
    )
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    fixture = Path(__file__).with_name("completion_pointer_race_main.inc")
    source = (
        args.source
        or Path(__file__).resolve().parents[4]
        / "src/plugins/gpunetio/gpunetio_kernels.cu"
    )
    text = source.read_text()
    old_guard = "if (DOCA_GPUNETIO_VOLATILE(completion_list[index].xferReqRingGpu) != nullptr) {"
    guard = old_guard if old_guard in text else "if (request != nullptr) {"
    if text.count(guard) != 1 or "#define ENABLE_DEBUG 0" not in text:
        parser.error("unsupported progress-kernel layout")
    text = text.replace(
        "#define ENABLE_DEBUG 0",
        "#define ENABLE_DEBUG 0\n__device__ uint32_t *test_gate;",
    )
    # Pause only after the pointer guard; the host then performs postXfer's reset.
    text = text.replace(
        guard,
        guard
        + """
                if (atomicCAS(test_gate, 0u, 1u) == 0) {
                    __threadfence_system();
                    while (*(volatile uint32_t *)(test_gate + 1) == 0) {}
                }
""",
    )
    output.mkdir(parents=True, exist_ok=True)
    driver = output / "completion_pointer_race.cu"
    driver.write_text(text + "\n" + fixture.read_text())
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("gpunetio_kernels.cu"))
    command = shlex.split(entry["command"])
    cleaned = []
    skip = False
    for arg in command:
        if skip:
            skip = False
        elif arg in ("-MF", "-MQ", "-MT"):
            skip = True
        elif arg not in ("-c", "-MD", "-MMD"):
            cleaned.append(
                str(driver)
                if arg == entry["file"]
                else (
                    str(output / "completion_pointer_race")
                    if arg == entry["output"]
                    else arg
                )
            )
    subprocess.run(cleaned, cwd=entry["directory"], check=True)
    logging.basicConfig(level=logging.INFO)
    logging.info("Built %s", output / "completion_pointer_race")


if __name__ == "__main__":
    main()
