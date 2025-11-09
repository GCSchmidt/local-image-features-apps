#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Unzip dataset
tar -xzf ./src/image-stitching/datasets/example-data.tgz -C ./src/image-stitching/datasets/