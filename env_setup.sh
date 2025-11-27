#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Installing Miniconda..."
    
    # For macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        bash Miniconda3-latest-MacOSX-x86_64.sh -b
        rm Miniconda3-latest-MacOSX-x86_64.sh
    # For Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b
        rm Miniconda3-latest-Linux-x86_64.sh
    else
        echo "For Windows users:"
        echo "1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
        echo "2. Install Miniconda following the instructions"
        echo "3. After installation, reopen this terminal and run this script again"
        exit 1
    fi
    
    export PATH="$HOME/miniconda3/bin:$PATH"
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Create conda environment
conda create -n imgext python=3.10 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate imgext

# Install requirements
pip install -r requirements.txt

echo "Environment 'imgext' has been set up successfully!"