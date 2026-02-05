#!/bin/bash
# Invoice OCR Extractor - Run Script
# ===================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Please install requirements:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run the application
echo "Starting Invoice OCR Extractor..."
echo "Open your browser to: http://localhost:8501"
echo ""

streamlit run app/main.py "$@"
