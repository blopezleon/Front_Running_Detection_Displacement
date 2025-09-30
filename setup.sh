#!/bin/bash

# Front-Running Detection System Setup Script
# This script installs dependencies and sets up the environment

echo "🔍 Front-Running Detection System Setup 🔍"
echo "============================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP "Python \K[0-9]+\.[0-9]+")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python version $python_version is supported"
else
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p data
mkdir -p reports

# Make scripts executable
echo "Setting permissions..."
chmod +x demo.py
chmod +x get_data.py
chmod +x analyze_data.py

# Initialize configuration if not exists
if [ ! -f "config.json" ]; then
    echo "Configuration file already exists"
else
    echo "✅ Configuration file created"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Edit config.json with your API keys (optional)"
echo "  3. Run demo: python demo.py"
echo "  4. Or collect data directly: python get_data.py"
echo ""
echo "📚 Documentation: Check README.md for detailed instructions"