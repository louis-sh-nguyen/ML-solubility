#!/usr/bin/env bash

# scripts/run_training.sh

set -e

echo "Starting training..."
python src/train.py
echo "Training complete."
