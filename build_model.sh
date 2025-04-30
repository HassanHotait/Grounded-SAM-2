#!/bin/bash

echo "🔥 Cleaning old temp files and __pycache__ folders..."
# Clean __pycache__ folders
find . -type d -name "__pycache__" -exec rm -r {} +

# Optional: also clean temp TorchServe folders if needed
rm -rf /tmp/grounded_sam2_florence2/

echo "✅ Clean complete."

echo "🚀 Archiving model into model_store/..."

# Run torch-model-archiver
torch-model-archiver \
  --model-name grounded_sam2_florence2 \
  --version 1.0 \
  --handler model_store/handler.py \
  --export-path model_store \
  --extra-files model_store/model.py,sam2/,checkpoints/,utils/ \
  --force

echo "✅ Model archive created: model_store/grounded_sam2_florence2.mar"

echo "🏁 Done!"
