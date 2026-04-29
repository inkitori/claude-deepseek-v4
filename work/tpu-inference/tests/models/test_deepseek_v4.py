# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Re-export of the V4 test suite at the path the autonomous-task spec
expects (`tests/models/test_deepseek_v4.py`).

The actual tests live alongside the rest of the JAX model tests at
`tests/models/jax/test_deepseek_v4.py` to keep the conftest import scoped
to the JAX-test package. This file re-imports them so `pytest
tests/models/test_deepseek_v4.py` collects the same set.
"""
import os
import sys
from pathlib import Path

# Force JAX to CPU before any JAX import — see DECISIONS.md D4.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=8")

# Make the JAX test module importable.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "jax"))

# Importing the module attaches its pytest test classes to this namespace.
from test_deepseek_v4 import *  # noqa: F401, F403
