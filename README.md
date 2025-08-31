# MetaphorScan

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Windows](https://img.shields.io/badge/platform-windows-lightgrey.svg)](https://www.microsoft.com/windows/)

**A standalone tool for detecting sedative and prophylactic metaphors in AI discourse**

MetaphorScan analyzes AI-related texts to identify metaphors that may obscure structural issues (sedative metaphors) or create false epistemic authority (prophylactic metaphors), providing precise replacements to promote epistemic clarity in AI research and discourse.

## 🎯 Overview

Based on the theoretical framework from:
- *The Nuremberg Defense of AI*
- *AI as an Epistemic Void Generator*  
- *The Alignment Problem as Epistemic Autoimmunity*

MetaphorScan detects and categorizes metaphors that shape how we understand AI systems, helping researchers and practitioners use more precise language when discussing AI capabilities and limitations.

## ✨ Key Features

- **🔍 Two-Stage Detection**: spaCy lexical matching + DistilBERT contextual validation
- **🌀 Attractor Basin Detection**: Identifies zones of high metaphor density
- **📊 Professional Reports**: Comprehensive PDF analysis with theoretical explanations
- **💻 Windows Optimized**: Built for MINGW64 environment with Python 3.12
- **📁 Multiple Formats**: Supports PDF, TXT, and Markdown input files
- **⚡ Large File Support**: Chunked processing for files up to 50MB

## 🚀 Quick Start

### Prerequisites
- Windows 11 with MINGW64 (Git Bash)
- Python 3.12+ (at `/c/Python312/python`)
- Virtual environment activated

### Installation & Usage

```bash
# Activate environment
envs/metaphorscan_env/Scripts/activate

# Basic analysis
python src/main.py --filepath data/raw/sample.txt

# With custom output
python src/main.py --filepath paper.pdf --output reports/my_analysis.pdf

# Verbose mode for detailed logging
python src/main.py --filepath document.txt --verbose
```

## 📈 Example Results

### Sample Analysis
**Input**: AI research text containing terms like "intelligence", "training", "hallucination"

**Output**:
- **12 metaphors detected** (3 sedative, 9 prophylactic)
- **86.3% average confidence**
- **2 epistemic attractor basins** identified
- **Professional PDF report** with explanations and recommendations

### Detected Metaphors
| Term | Category | Replacement | Theoretical Basis |
|------|----------|-------------|-------------------|
| "hallucination" | Sedative | "fabrication" | *Epistemic Autoimmunity*, Section 2 |
| "intelligence" | Prophylactic | "pattern simulation" | *Epistemic Void Generator* |
| "training" | Prophylactic | "data conditioning" | *Nuremberg Defense* |
| "alignment" | Prophylactic | "structure-tracking" | *Epistemic Autoimmunity*, Section 3 |

## 📖 Documentation

- **[Complete User Guide](docs/README.md)** - Detailed installation, usage, and examples
- **[API Documentation](docs/api_docs.md)** - Developer reference and technical details
- **[User Manual](docs/user_guide.md)** - Step-by-step usage instructions

## 🏗️ Architecture

```
MetaphorScan/
├── src/
│   ├── main.py                    # CLI application entry point
│   ├── pipeline/                  # Two-stage analysis pipeline
│   ├── text_processing/           # Input file processing
│   ├── output/                    # Report generation
│   └── config/                    # Lexicon and settings
├── data/                          # Models and input files
├── outputs/                       # Generated reports and logs
└── docs/                          # Documentation
```

## 🧪 Testing

The application has been tested with various AI texts and academic papers:

```bash
# Test with sample data
python src/main.py --filepath data/raw/ai_research_sample.txt --output outputs/reports/test_report.pdf
```

**Test Results**:
- ✅ Successfully processes PDF and text files
- ✅ Detects metaphors with high accuracy (85-95% confidence)
- ✅ Generates professional PDF reports
- ✅ Handles large files with chunked processing
- ✅ Provides detailed logging and error handling

## 🤝 Contributing

We welcome contributions that advance epistemic clarity in AI discourse:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/Soyuz43/MetaphorScan.git
cd MetaphorScan

# Set up environment
bash envs/setup_env.sh

# Install development dependencies
pip install -r envs/dev_requirements.txt

# Run tests
bash scripts/run_tests.sh
```

## 📋 Requirements

### System Requirements
- **OS**: Windows 11 (MINGW64 environment)
- **Python**: 3.12+
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 2GB free space

### Dependencies
- spaCy 3.8+ with English model
- Transformers 4.56+ (DistilBERT)
- PyPDF2 for PDF processing
- ReportLab for PDF generation
- scikit-learn for similarity calculations

## 🔒 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use MetaphorScan in your research, please cite:

```bibtex
@software{metaphorscan2024,
  title={MetaphorScan: Detecting Sedative and Prophylactic Metaphors in AI Discourse},
  author={MetaphorScan Contributors},
  year={2024},
  url={https://github.com/Soyuz43/MetaphorScan}
}
```

## 🎓 Theoretical Background

This tool implements analysis methods described in:
- **The Nuremberg Defense of AI**: How anthropomorphic metaphors deflect responsibility
- **AI as an Epistemic Void Generator**: Technical terminology creating false precision  
- **The Alignment Problem as Epistemic Autoimmunity**: Metaphors masking structural issues

## 📞 Support

- **Documentation**: See [docs/README.md](docs/README.md) for detailed guides
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join conversations about AI discourse clarity

## 🌟 Acknowledgments

- spaCy team for robust NLP processing
- Hugging Face for DistilBERT model access
- ReportLab for PDF generation capabilities
- The authors of the foundational theoretical papers

---

**MetaphorScan** - Promoting epistemic clarity in AI discourse through systematic metaphor detection and analysis.
