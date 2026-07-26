#!/usr/bin/env python3
"""Benchmark simulator, RGB global pass, ingest, and rollout locally."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from world_model.evaluation.latency import benchmark_callable
from world_model.observations import ObservationContext, ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.utils.config import load_config
from world_model.utils.device import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    overrides = [f"device.preference={args.device}"] if args.device else []
    config = load_config(args.config, overrides=overrides)
    device = select_device(config.device.preference)
    start = time.perf_counter()
    episode = generate_episode(config, config.project.seed)
    simulator_ms = (time.perf_counter() - start) * 1000
    model = OnlineWorldModel.from_config(config, device=device.device).eval()
    packet = ObservationPacket(
        modality="rgb",
        sensor_id="camera0",
        timestamp=float(episode["timestamps"][0]),
        payload=episode["rgb"][0].to(device.device),
        calibration={
            "intrinsics": episode["camera"]["intrinsics"][0].to(device.device),
            "world_from_camera": episode["camera"]["world_from_camera"][0].to(device.device),
        },
        frame_id="camera:camera0",
        metadata={"image_size": config.simulator.image_size},
    )
    context = ObservationContext(
        timestamp=packet.timestamp,
        calibration=packet.calibration,
        frame_id=packet.frame_id,
        max_objects=config.model.max_objects,
        device=device.device,
        dtype=torch.float32,
        training=False,
        metadata=packet.metadata,
    )
    rgb_module = model.observation_modules["rgb"]
    with torch.no_grad():
        global_latency = benchmark_callable(
            lambda: rgb_module.initialise_measurements([packet], context),
            device=device.device,
            warmup=config.evaluation.benchmark_warmup,
            repeats=args.repeats,
        )

        def first_ingest() -> None:
            model.reset()
            model.ingest(packet)

        ingest_latency = benchmark_callable(
            first_ingest,
            device=device.device,
            warmup=1,
            repeats=args.repeats,
        )
        model.reset()
        model.ingest(packet)
        rollout_latency = benchmark_callable(
            lambda: model.predict([0.1, 0.5, 1.0]),
            device=device.device,
            warmup=1,
            repeats=args.repeats,
        )
    report = {
        "device": str(device.device),
        "torch": device.torch_version,
        "mps_available": device.mps_available,
        "image_size": config.simulator.image_size,
        "max_objects": config.model.max_objects,
        "episode_generation_ms": simulator_ms,
        "rgb_global": global_latency,
        "first_ingest": ingest_latency,
        "rollout_0.1_0.5_1.0": rollout_latency,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
