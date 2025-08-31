# MetaphorScan

A standalone tool to detect sedative and prophylactic metaphors in AI texts, inspired by *The Nuremberg Defense of AI*, *AI as an Epistemic Void Generator*, and *The Alignment Problem as Epistemic Autoimmunity*. 

MetaphorScan analyzes AI-related texts to identify metaphors that may obscure structural issues (sedative metaphors like "hallucination") or create false epistemic authority (prophylactic metaphors like "intelligence"), providing precise replacements to promote epistemic clarity.

## Features

- **Two-Stage Detection Pipeline**: Lexical matching with spaCy + DistilBERT contextual validation
- **Epistemic Attractor Basin Detection**: Identifies high-density metaphor clusters that create zones of conceptual confusion
- **Comprehensive PDF Reports**: Professional analysis reports with highlighted metaphors and theoretical explanations
- **Windows Compatible**: Optimized for MINGW64 environment with Python 3.12
- **Large File Support**: Chunked processing for files up to 50MB
- **Multiple Formats**: Supports PDF, TXT, and MD input files

## Quick Start

### 1. Environment Setup
```bash
# Activate the virtual environment (Windows MINGW64)
envs/metaphorscan_env/Scripts/activate

# Install dependencies (if not already installed)
pip install spacy transformers PyPDF2 python-docx PyYAML pandas reportlab numpy torch datasets scikit-learn

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Basic Usage
```bash
# Analyze a text file with default settings
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath data/raw/sample.txt

# Analyze a PDF with custom output location
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath data/raw/paper.pdf --output outputs/reports/my_analysis.pdf

# Enable verbose logging for detailed output
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath data/raw/sample.txt --verbose
```

## Example Usage

### Example 1: Basic Text Analysis
```bash
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath data/raw/sample.txt --output outputs/reports/test_report.pdf --verbose
```

**Sample Input** (`data/raw/sample.txt`):
```
The artificial intelligence model demonstrated remarkable learning capabilities during extensive training on large datasets. However, the system occasionally hallucinated outputs that diverged significantly from expected patterns. This error rate suggests fundamental alignment challenges in human-AI interaction paradigms.
```

**Analysis Results**:
- **10 metaphors detected** (1 sedative, 9 prophylactic)
- **87.4% average confidence**
- **8 epistemic attractor basins** detected
- **100% validation rate**

### Example 2: AI Research Paper Analysis
```bash
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath data/raw/ai_research_sample.txt --output outputs/reports/ai_research_report.pdf
```

**Sample Input** (`data/raw/ai_research_sample.txt`):
```
Machine learning algorithms demonstrate intelligent behavior through sophisticated training regimens. When these systems hallucinate false information, researchers must address alignment challenges to ensure responsible AI development.
```

**Analysis Results**:
- **12 metaphors detected** (3 sedative, 9 prophylactic)
- **86.3% average confidence**
- **2 epistemic attractor basins** detected
- **100% validation rate**

### Example 3: Large File with Chunked Processing
```bash
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath large_paper.pdf --chunk-size 2000 --verbose
```

### Example 4: Analysis Only (No PDF Report)
```bash
envs/metaphorscan_env/Scripts/activate && python src/main.py --filepath article.txt --no-report
```

## Command Line Options

```bash
usage: main.py [-h] --filepath FILEPATH [--output OUTPUT] [--chunk-size CHUNK_SIZE] 
               [--verbose] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--no-report]

options:
  --filepath FILEPATH   Path to input file (PDF, TXT, MD)
  --output OUTPUT       Path to output PDF report (default: outputs/reports/metaphorscan_report.pdf)
  --chunk-size CHUNK_SIZE
                        Chunk size for large file processing (default: auto-detect)
  --verbose, -v         Enable verbose logging output
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set logging level (default: INFO)
  --no-report           Skip PDF report generation (analysis only)
```

## Detected Metaphor Categories

### Sedative Metaphors
Terms that obscure structural AI issues by framing failures as natural phenomena:
- **"hallucination"** → **"fabrication"** (*Epistemic Autoimmunity*, Section 2)
- **"error"** → **"fabrication"** (*Nuremberg Defense*)

### Prophylactic Metaphors  
Terms that create false epistemic authority through anthropomorphic analogies:
- **"intelligence"** → **"pattern simulation"** (*Epistemic Void Generator*)
- **"training"** → **"data conditioning"** (*Nuremberg Defense*)
- **"alignment"** → **"structure-tracking"** (*Epistemic Autoimmunity*, Section 3)

## Output Reports

MetaphorScan generates comprehensive PDF reports containing:

1. **Executive Summary**: Overview of detected metaphors and attractor basins
2. **Detailed Analysis**: Table of flagged metaphors with confidence scores and explanations
3. **Attractor Basin Detection**: Areas of high metaphor density indicating epistemic confusion
4. **Highlighted Text**: Original text with metaphor annotations
5. **Theoretical Framework**: Connections to critical AI literature
6. **Recommendations**: Suggested improvements for epistemic clarity

## Theoretical Framework

Based on three foundational papers:

- **The Nuremberg Defense of AI**: Analysis of how anthropomorphic metaphors deflect responsibility
- **AI as an Epistemic Void Generator**: Examination of how technical terminology creates false precision
- **The Alignment Problem as Epistemic Autoimmunity**: Study of how metaphors mask structural AI issues

## System Requirements

- **Operating System**: Windows 11 (MINGW64 environment)
- **Python**: 3.12+ (located at `/c/Python312/python`)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large files)
- **Storage**: 2GB free space for models and dependencies

## Project Structure

```
MetaphorScan/
├── src/
│   ├── main.py                    # CLI entry point
│   ├── pipeline/                  # Analysis pipeline
│   │   ├── lexical_matcher.py     # spaCy-based lexical detection
│   │   ├── contextual_analyzer.py # DistilBERT validation
│   │   └── pipeline.py            # Pipeline orchestration
│   ├── text_processing/           # Input processing
│   │   ├── extract.py             # PDF/text extraction
│   │   └── preprocess.py          # Text normalization
│   ├── output/                    # Report generation
│   │   └── report_generator.py    # PDF report creation
│   └── config/                    # Configuration
│       ├── lexicon.yaml           # Metaphor definitions
│       └── settings.yaml          # Pipeline settings
├── data/                          # Data storage
│   ├── raw/                       # Input files
│   ├── models/                    # Downloaded models
│   └── processed/                 # Intermediate results
├── outputs/                       # Generated outputs
│   ├── reports/                   # PDF reports
│   └── logs/                      # Analysis logs
└── envs/                          # Environment setup
    └── metaphorscan_env/          # Virtual environment
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated:
   ```bash
   envs/metaphorscan_env/Scripts/activate
   ```

2. **Model Download Failures**: Check internet connection and run:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **PDF Generation Issues**: Verify reportlab installation:
   ```bash
   pip install reportlab
   ```

4. **Large File Processing**: Use chunked processing:
   ```bash
   python src/main.py --filepath large_file.pdf --chunk-size 1000
   ```

### Log Files

Detailed logs are saved to `outputs/logs/` with timestamps for debugging:
```
outputs/logs/metaphorscan_20250831_130507.log
```

## Contributing

For development and API documentation, see:
- `docs/user_guide.md` - Detailed usage instructions
- `docs/api_docs.md` - Developer documentation and API reference

## License

This project implements critical analysis of AI discourse as described in the referenced academic papers. Use responsibly to promote epistemic clarity in AI research and development.