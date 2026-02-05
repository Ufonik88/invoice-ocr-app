# Invoice OCR Extractor

A desktop application for extracting structured data from South African tax invoices using OCR and LLM technology.

## Features

- 📄 **Multi-format support**: Process PDFs and images (PNG, JPG, TIFF, BMP)
- 🔍 **Tesseract OCR**: High-quality text extraction with preprocessing options
- 🤖 **Multi-LLM support**: Works with Ollama, LM Studio (local), and Deepseek (cloud)
- 👁️ **Vision models**: Direct image extraction with llava (skip OCR)
- 💰 **ZAR formatting**: South African Rand currency and 15% VAT handling
- 📊 **Excel export**: Each line item as a separate row with shared header data
- ✏️ **Editable results**: Review and correct extracted data before export

## Prerequisites

### 1. Python 3.9+

Ensure you have Python 3.9 or later installed:

```bash
python --version
```

### 2. Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Poppler (for PDF support)

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt install poppler-utils
```

**Windows:**
Download from [poppler releases](https://github.com/osber/pdf2image/wiki/Windows)

### 4. LLM Provider (at least one)

**Option A: Ollama (Recommended for local use)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# For vision support (optional)
ollama pull llava
```

**Option B: LM Studio**
1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Load a model
3. Start the local server

**Option C: Deepseek (Cloud)**
1. Get API key from [deepseek.com](https://www.deepseek.com/)
2. Set in `.env` file

## Installation

1. **Clone or download this project**

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment (optional):**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Verify setup:**
```bash
python scripts/check_setup.py
```

## Usage

### Running the Application

```bash
streamlit run app/main.py
```

Or use the convenience script:
```bash
./scripts/run_app.sh
```

### Workflow

1. **Upload Invoice**: Drag and drop a PDF or image file
2. **Configure Settings**: Select LLM provider and OCR options in sidebar
3. **Extract Data**: Click "Run OCR + LLM Extraction" or use vision model
4. **Review & Edit**: Verify extracted data and make corrections
5. **Export**: Save to Excel file

### Supported Invoice Fields

- **Header**: Invoice number, date, due date, PO number
- **Seller**: Name, address, VAT number, contact info
- **Customer**: Name, account number, address
- **Line Items**: Description, quantity, unit price, total, VAT rate
- **Totals**: Subtotal, VAT amount, total due, discounts

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Default LLM Provider
DEFAULT_LLM_PROVIDER=ollama  # ollama, lm_studio, or deepseek

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_VISION_MODEL=llava

# LM Studio Settings
LM_STUDIO_BASE_URL=http://localhost:1234

# Deepseek Settings (for cloud fallback)
DEEPSEEK_API_KEY=your-api-key-here

# Tesseract Settings
TESSERACT_CMD=tesseract  # Full path if not in PATH
```

### OCR Preprocessing Levels

- **None**: No preprocessing (fastest)
- **Light**: Grayscale + contrast enhancement
- **Standard**: Recommended - includes noise reduction and thresholding
- **Aggressive**: Full preprocessing with deskew (for poor quality scans)

## Project Structure

```
invoice_ocr_app/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── main.py            # Streamlit application
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── preprocessor.py  # Image preprocessing
│   │   └── extractor.py     # OCR extraction
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py       # LLM prompts
│   │   ├── client.py        # Multi-provider client
│   │   └── parser.py        # Response parsing
│   ├── models/
│   │   ├── __init__.py
│   │   └── invoice.py       # Data models
│   ├── export/
│   │   ├── __init__.py
│   │   └── excel.py         # Excel export
│   └── ui/
│       └── __init__.py
├── scripts/
│   ├── check_setup.py       # Setup validation
│   └── run_app.sh           # Run script
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Troubleshooting

### Tesseract not found
- Verify installation: `tesseract --version`
- Set full path in `.env`: `TESSERACT_CMD=/opt/homebrew/bin/tesseract`

### Ollama connection refused
- Start Ollama: `ollama serve`
- Check status: `curl http://localhost:11434/api/tags`

### PDF extraction fails
- Install Poppler (see prerequisites)
- On Windows, add to PATH

### Poor OCR quality
- Try "Aggressive" preprocessing level
- Ensure image is at least 300 DPI
- Use vision model if available

## License

MIT License - feel free to modify and distribute.

## Contributing

Contributions welcome! Please submit issues and pull requests.
