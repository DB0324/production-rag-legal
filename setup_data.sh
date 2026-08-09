#!/bin/bash
# =============================================================
# Unzip data_transfer.zip on the remote server.
# Run from project root after uploading the zip file.
#
# Usage:
#   cd ~/Dhara-Mtech/IPR/PROJECTS/production-rag-legal
#   bash setup_data.sh
# =============================================================
set -e

echo "=============================================="
echo "  Unzipping data_transfer.zip"
echo "=============================================="

if [ ! -f "data_transfer.zip" ]; then
    echo "ERROR: data_transfer.zip not found in current directory."
    exit 1
fi

# Unzip (overwrites existing files)
echo "Unzipping..."
unzip -o data_transfer.zip

# Create logs directory
mkdir -p logs

# Verify critical files
echo ""
echo "Verifying files..."
python -m src.ingestion.verify_data

echo ""
echo "Done! Next: bash run_ablation.sh"
