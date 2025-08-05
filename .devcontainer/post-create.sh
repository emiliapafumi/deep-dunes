#!/bin/bash

# Post-create script for Deep Dunes OTBTF devcontainer
echo "🚀 Setting up Deep Dunes development environment..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install Python requirements
echo "🐍 Installing Python requirements..."
cd /workspaces/deep-dunes
pip install --upgrade pip

# Check existing installations first
pip install -r scripts/requirements.txt

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p deep-dunes-data
mkdir -p models/logs
mkdir -p models/output

# Display installed packages
# echo "📋 Installed Python packages:"
# pip list

echo "✅ Setup complete! You can now:"
echo "   • Run sampling: python scripts/1-sampling.py --help"
echo "   • Run training: python scripts/2-training.py --help"  
echo "   • Run inference: python scripts/3-inference.py --help"
echo ""
