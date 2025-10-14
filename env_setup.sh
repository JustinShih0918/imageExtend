#!/bin/bash

# Create conda environment
conda create -n imgext python=3.10 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imgext

# Install requirements
pip install -r requirements.txt

echo "Environment 'imgext' has been set up successfully!"