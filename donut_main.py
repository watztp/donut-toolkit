#!/usr/bin/env python3
"""Backward-compatible wrapper for the training script.

New code should use ``python -m scripts.train`` directly.
"""

from scripts.train import main


if __name__ == "__main__":
    main()
