#!/bin/bash

set -e # ensures the script exits immediately if any command fails.

# Set variables
STUDIO_DIR="/teamspace/studios/this_studio"
KAGGLE_DIR="$STUDIO_DIR/.kaggle"

# Download ImageNet 1M using Kaggle API
pip install kaggle
mkdir -p data
cd /cache || exit

# Ensure kaggle.json permission is correct
chmod 600 "$KAGGLE_DIR/kaggle.json"

# Download and unzip dataset
kaggle competitions download -c imagenet-object-localization-challenge
unzip -qq imagenet-object-localization-challenge.zip '*.JPEG' -d ./data

# Copy dataset to desired location
cp -r /cache/data/ILSVRC/Data/CLS-LOC/* "$STUDIO_DIR/data/" 2>/dev/null

# Remove main.py if exists
rm -f main.py

# remove the zip file and the extracted folder
rm -f imagenet-object-localization-challenge.zip
rm -rf /cache/data

# Output elapsed time
echo "Process completed in $SECONDS seconds"
