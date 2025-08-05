#!/bin/bash

# Post-create script for Deep Dunes OTBTF devcontainer
echo "🚀 Setting up Deep Dunes development environment..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install additional useful packages
echo "🔧 Installing additional development tools..."
apt-get install -y \
    vim \
    nano \
    htop \
    tree \
    curl \
    wget \
    unzip \
    git-lfs

# Install Python requirements
echo "🐍 Installing Python requirements..."
cd /workspaces/deep-dunes
pip install --upgrade pip

# Check existing installations first
echo "🔍 Checking existing package versions..."
echo "Current Python packages:"
pip list | grep -E "(tensorflow|tensorboard|keras|otbtf)" || echo "None of the ML packages found"

# Check if TensorFlow is already installed and working
TF_INSTALLED=$(python -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ TensorFlow $TF_INSTALLED is already installed"
    TB_INSTALLED=$(python -c "import tensorboard; print(tensorboard.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "✅ TensorBoard $TB_INSTALLED is already installed"
    fi
    
    # Install only non-conflicting packages
    echo "📦 Installing additional packages..."
    pip install pandas geopandas rasterio scikit-learn numpy gdown
    
    # Try to install otbtf (should be compatible)
    pip install otbtf || echo "⚠️  OTBTF installation failed - it might already be available"
else
    echo "🔎 TensorFlow not found, installing from requirements..."
    pip install -r scripts/requirements.txt
fi

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p models/logs
mkdir -p models/output

# Set up git (if not already configured)
echo "📝 Setting up git configuration..."
if [ ! -f ~/.gitconfig ]; then
    git config --global init.defaultBranch main
    git config --global core.autocrlf input
    git config --global pull.rebase false
fi

# Display installed packages
# echo "📋 Installed Python packages:"
# pip list

echo "✅ Setup complete! You can now:"
echo "   • Run sampling: python scripts/1-sampling.py --help"
echo "   • Run training: python scripts/2-training.py --help"  
echo "   • Run inference: python scripts/3-inference.py --help"
echo ""
