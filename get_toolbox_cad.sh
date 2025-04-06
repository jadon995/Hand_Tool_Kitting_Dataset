#!/bin/bash

# Exit on error
set -e

# Define variables
FILE_ID="1IrCZrsqz8jZiyXAX9Rhp89BfYuX4KPdY"
OUTPUT_NAME="toolbox_cad.zip"
TARGET_DIR="assets"

# Create assets directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing it now..."
    pip install gdown
fi

# Download into assets/
echo "Downloading into $TARGET_DIR..."
gdown https://drive.google.com/uc?id=$FILE_ID -O "$TARGET_DIR/$OUTPUT_NAME"

# Extract the zip file in the assets folder
echo "Extracting $OUTPUT_NAME into $TARGET_DIR..."
unzip -o "$TARGET_DIR/$OUTPUT_NAME" -d "$TARGET_DIR"

echo "✅ Download and extraction completed in $TARGET_DIR."
