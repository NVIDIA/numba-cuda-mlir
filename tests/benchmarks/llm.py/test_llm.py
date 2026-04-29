#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import subprocess
import sys
from pathlib import Path

import pytest

from cusimt import tools

LLM_DIR = Path(__file__).parent
cc = tools.get_gpu_compute_capability(tuple)
cuda_version = tools.get_cuda_runtime_version()

if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))


@pytest.fixture(scope="module")
def setup_llm_data():
    """Run prepro_tinyshakespeare.py and train_gpt2.py to prepare data and model."""
    data_exists = (LLM_DIR / "data" / "tiny_shakespeare_train.bin").exists() and (
        LLM_DIR / "data" / "tiny_shakespeare_val.bin"
    ).exists()
    model_exists = (LLM_DIR / "gpt2_124M.bin").exists()

    if not data_exists:
        prepro_script = LLM_DIR / "prepro_tinyshakespeare.py"
        subprocess.run([sys.executable, str(prepro_script)], cwd=LLM_DIR, check=True)

    if not model_exists:
        train_script = LLM_DIR / "train_gpt2.py"
        subprocess.run([sys.executable, str(train_script)], cwd=LLM_DIR, check=True)

    yield LLM_DIR


@pytest.mark.skipif(cuda_version < (13, 0), reason="Requires CUDA toolkit 13.0 or newer")
@pytest.mark.xfail(reason="CCCL Support is incomplete")
def test_llm_training_loss(setup_llm_data):
    """Test that LLM training produces valid losses (< 5 and not NaN)."""
    from llm import run_training

    working_dir = setup_llm_data

    train_losses, val_losses = run_training(
        input_dataset_prefix="data/tiny_shakespeare",
        B=4,
        T=1024,
        learning_rate=1e-4,
        val_loss_every=20,
        val_max_batches=20,
        sample_every=20,
        genT=64,
        working_dir=working_dir,
    )

    for i, loss in enumerate(train_losses):
        assert not math.isnan(loss), f"Train loss at step {i} is NaN"
        assert loss < 5, f"Train loss at step {i} is {loss}, expected < 5"

    for i, loss in enumerate(val_losses):
        assert not math.isnan(loss), f"Val loss at step {i} is NaN"
        assert loss < 5, f"Val loss at step {i} is {loss}, expected < 5"

    assert len(train_losses) > 0, "No training losses recorded"
    assert len(val_losses) > 0, "No validation losses recorded"


if __name__ == "__main__":
    import os

    os.chdir(LLM_DIR)

    data_exists = (LLM_DIR / "data" / "tiny_shakespeare_train.bin").exists() and (
        LLM_DIR / "data" / "tiny_shakespeare_val.bin"
    ).exists()
    model_exists = (LLM_DIR / "gpt2_124M.bin").exists()

    if not data_exists:
        prepro_script = LLM_DIR / "prepro_tinyshakespeare.py"
        subprocess.run([sys.executable, str(prepro_script)], cwd=LLM_DIR, check=True)

    if not model_exists:
        train_script = LLM_DIR / "train_gpt2.py"
        subprocess.run([sys.executable, str(train_script)], cwd=LLM_DIR, check=True)

    from llm import run_training

    train_losses, val_losses = run_training(
        input_dataset_prefix="data/tiny_shakespeare",
        B=4,
        T=1024,
        learning_rate=1e-4,
        val_loss_every=20,
        val_max_batches=20,
        sample_every=20,
        genT=64,
    )

    for i, loss in enumerate(train_losses):
        assert not math.isnan(loss), f"Train loss at step {i} is NaN"
        assert loss < 5, f"Train loss at step {i} is {loss}, expected < 5"

    for i, loss in enumerate(val_losses):
        assert not math.isnan(loss), f"Val loss at step {i} is NaN"
        assert loss < 5, f"Val loss at step {i} is {loss}, expected < 5"

    print("All loss checks passed!")
