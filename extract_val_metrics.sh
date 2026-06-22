#!/bin/bash
# Extract validation metrics from flux output files
# Usage: ./extract_val_metrics.sh flux-*.out

echo "Epoch,valid_rate"
for f in "$@"; do
  grep -o "Epoch [0-9]*: 100%.*val_qm9/valid_rate=[0-9.]*" "$f" 2>/dev/null | \
  sed 's/.*Epoch \([0-9]*\):.* val_qm9\/valid_rate=\([0-9.]*\).*/\1,\2/' | \
  sort -n -u
done
