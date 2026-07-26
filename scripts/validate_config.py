#!/usr/bin/env python3
"""Resolve and validate an Orpheus YAML file without running a model."""

from __future__ import annotations

import argparse

import yaml

from world_model.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.set)
    print(yaml.safe_dump(config.to_dict(), sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
