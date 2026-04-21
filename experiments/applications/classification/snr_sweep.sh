#!/usr/bin/env bash
# Run Exp 3 at multiple SNR levels sequentially.
# Each run writes to its own results dir and log.
set -euo pipefail
cd "$(dirname "$0")/../../.."

for SNR in 10 20; do
    echo "=== Starting SNR=${SNR} dB ==="
    PYTHONPATH=. python3 -u experiments/applications/classification/run.py \
        --n-train 1000 --n-val 200 --n-test 200 \
        --n-samples 1024 --sample-rate 1000.0 \
        --epochs-classical 50 --epochs-nemd 50 \
        --snr-db "${SNR}.0" \
        --out-dir "paper/figures/phase3_exp3_snr${SNR}" \
        > "/tmp/nemd_exp3_snr${SNR}.log" 2>&1
    echo "=== Done SNR=${SNR} dB ==="
done
echo "=== Sweep complete ==="
