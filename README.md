# Vectory

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Precision LLM Evaluation** - A powerful, professional-grade toolkit for evaluating Large Language Model outputs.

## Features

- **📊 Dataset Management** - Upload JSON, CSV, or PDF files. Automatic column detection and mapping.
- **🤖 LLM-as-Judge** - Use GPT-4 or Claude to automatically evaluate output quality with customizable criteria.
- **📏 Rule-Based Metrics** - Calculate BLEU, ROUGE, exact match, Levenshtein similarity, and regex patterns.
- **👤 Human Evaluation** - Rate outputs manually with customizable scales and criteria.
- **🏆 MTEB Leaderboard** - View embedding model benchmarks from the Massive Text Embedding Benchmark.
- **🧪 Custom Benchmarks** - Run your own embedding evaluations using the MTEB framework.
- **⚙️ Configurable Themes** - Switch between Dark, Light, and Colorblind-safe themes for accessibility.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tacticaledge/vectory-ai.git
cd vectory-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Standard Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for some metrics)
python -c "import nltk; nltk.download('punkt')"
```

### Optional: MTEB Support

For embedding evaluation features (Leaderboard & Custom Benchmarks), you need PyTorch >= 2.1.0:

```bash
# Install PyTorch (see https://pytorch.org for platform-specific instructions)
pip install torch>=2.1.0

# Then uncomment MTEB dependencies in requirements.txt and install
pip install mteb sentence-transformers
```

## Usage

### 1. Upload Your Data

Navigate to the **📊 Dataset** page and upload your evaluation data:

- **JSON/JSONL**: Arrays of objects with input/output/expected fields
- **CSV**: Tabular data with headers
- **PDF**: Extracted to markdown (optimized for LLM consumption)

The app will automatically detect column roles, or you can map them manually.

### 2. Run Evaluations

Choose your evaluation method:

#### LLM-as-Judge (🤖)
- Select provider (OpenAI or Anthropic)
- Enter your API key
- Choose evaluation criteria
- Run batch evaluation

#### Rule-Based Metrics (📏)
- Select metrics (BLEU, ROUGE, exact match, etc.)
- Run evaluation
- View per-sample and aggregate results

#### Human Evaluation (👤)
- Navigate through samples
- Rate on customizable scales
- Add feedback and criteria tags
- Export annotations

### 3. Export Results

Download your evaluation results as CSV or JSON from any results page.

## Configuration

### API Keys

Configure API keys in one of three ways:

#### Option 1: Streamlit Secrets (Recommended for deployment)

Create `.streamlit/secrets.toml`:

```toml
[api_keys]
openai = "sk-..."
anthropic = "sk-ant-..."
```

#### Option 2: Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Option 3: Manual Input

Enter your API key directly in the UI (stored only in session).

### Themes & Accessibility

Vectory includes multiple themes to suit different preferences and accessibility needs. Change your theme in the **⚙️ Settings** page.

#### Available Themes:

| Theme | Description |
|-------|-------------|
| **Dark Developer** | Modern dark theme optimized for developers (default) |
| **Light** | Clean light theme for bright environments |
| **High Contrast (Colorblind Safe)** | Blue-orange palette distinguishable by most forms of color blindness |
| **High Contrast Dark** | Maximum contrast for users with low vision |

The colorblind-safe theme uses a carefully selected palette that works for:
- Deuteranopia (red-green color blindness)
- Protanopia (red color blindness)
- Tritanopia (blue-yellow color blindness)

### Streamlit Configuration

The default theme is configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
font = "monospace"
```

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Manual Deployment

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Project Structure

```
vectory/
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── pages/                    # Streamlit pages
│   ├── 0_⚙️_Settings.py     # Theme & preferences
│   ├── 1_📊_Dataset.py      # Dataset upload & config
│   ├── 2_🤖_LLM_Judge.py    # LLM-as-Judge evaluation
│   ├── 3_📏_Metrics.py      # Rule-based metrics
│   ├── 4_👤_Human_Eval.py   # Human evaluation
│   ├── 5_🏆_Leaderboard.py  # MTEB leaderboard
│   └── 6_🧪_Custom_Benchmark.py
├── components/               # Core modules
│   ├── __init__.py          # Version and branding
│   ├── models.py            # Pydantic data models
│   ├── themes.py            # Configurable theme system
│   ├── document_loader.py   # File loading utilities
│   ├── ui.py                # Animated UI components
│   ├── mteb_evaluator.py    # Embedding evaluation
│   └── evaluators/          # Evaluation implementations
│       ├── llm_judge.py
│       └── metrics.py
├── tests/                    # Test suite
└── data/                     # Sample datasets
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=components --cov-report=html
```

## Troubleshooting

### PyTorch Version Error

If you see `module 'torch' has no attribute 'compiler'`:

```bash
pip install torch>=2.1.0
```

The app will work without PyTorch, but MTEB features will be disabled.

### NLTK Data Missing

If you see NLTK-related errors:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### API Key Issues

- Ensure your API key is valid and has sufficient credits
- Check that the key is correctly formatted (no extra spaces)
- For Anthropic, ensure you're using the correct model names

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) - The amazing framework powering this app
- [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark
- [PyMuPDF4LLM](https://pymupdf.readthedocs.io/) - PDF to Markdown conversion
- [HuggingFace Evaluate](https://huggingface.co/docs/evaluate/) - Evaluation metrics

---

**Vectory** - Precision LLM Evaluation | [Tactical Edge](https://tacticaledgeai.com)
