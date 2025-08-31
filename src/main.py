# MetaphorScan/src/main.py
"""
MetaphorScan CLI application for detecting sedative and prophylactic metaphors.
Integrates text extraction, two-stage analysis pipeline, and PDF report generation.

Implements comprehensive metaphor detection framework from *The Nuremberg Defense of AI*,
*AI as an Epistemic Void Generator*, and *The Alignment Problem as Epistemic Autoimmunity*.
"""
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from pipeline.pipeline import run_pipeline
from output.report_generator import generate_metaphor_report

# Configure logging
def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration for the application."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"metaphorscan_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"MetaphorScan started - Log file: {log_file}")
    return logger

def validate_input_file(filepath: str) -> Path:
    """
    Validate input file exists and is supported format.
    
    Args:
        filepath: Path to input file
        
    Returns:
        Path: Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {filepath}")
    
    # Check file extension
    supported_extensions = {'.pdf', '.txt', '.md', '.text'}
    if file_path.suffix.lower() not in supported_extensions:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. "
                        f"Supported formats: {', '.join(supported_extensions)}")
    
    return file_path

def validate_output_path(output_path: str) -> Path:
    """
    Validate and prepare output path for report generation.
    
    Args:
        output_path: Desired output file path
        
    Returns:
        Path: Validated output Path object
    """
    output_file = Path(output_path)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure PDF extension
    if output_file.suffix.lower() != '.pdf':
        output_file = output_file.with_suffix('.pdf')
    
    return output_file

def print_analysis_summary(results: dict, logger: logging.Logger):
    """Print a summary of analysis results to console."""
    stats = results.get('statistics', {})
    
    print("\n" + "="*60)
    print("          METAPHORSCAN ANALYSIS COMPLETE")
    print("="*60)
    print(f"Input file: {results.get('metadata', {}).get('input_file', 'Unknown')}")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    # Core statistics
    print("METAPHOR DETECTION RESULTS:")
    print(f"  Total metaphors detected: {stats.get('total_metaphors', 0)}")
    print(f"  • Sedative metaphors: {stats.get('sedative_count', 0)}")
    print(f"  • Prophylactic metaphors: {stats.get('prophylactic_count', 0)}")
    print(f"  Average confidence: {stats.get('average_confidence', 0):.1%}")
    print(f"  Validation rate: {stats.get('validation_rate', 0):.1%}")
    
    # Attractor basins
    basin_count = stats.get('attractor_basin_count', 0)
    print(f"\nEPISTEMIC ATTRACTOR BASINS:")
    print(f"  Total basins detected: {basin_count}")
    if basin_count > 0:
        print(f"  • Lexical basins: {stats.get('lexical_basin_count', 0)}")
        print(f"  • Semantic basins: {stats.get('semantic_basin_count', 0)}")
    
    # Text statistics
    print(f"\nTEXT ANALYSIS:")
    print(f"  Text length: {stats.get('text_length', 0):,} characters")
    print(f"  Sentences: {stats.get('sentence_count', 0)}")
    print(f"  Metaphor density: {stats.get('metaphor_density', 0):.3f} metaphors/sentence")
    
    # Detailed metaphor list
    validated_matches = results.get('validated_matches', [])
    if validated_matches:
        print(f"\nDETECTED METAPHORS:")
        for i, match in enumerate(validated_matches, 1):
            term = match.get('term', '')
            category = match.get('category', '')
            replacement = match.get('replacement', '')
            confidence = match.get('confidence', 0)
            print(f"  {i}. '{term}' ({category}) → '{replacement}' [{confidence:.1%}]")
    
    # Warnings
    if basin_count > 2:
        print(f"\n⚠ WARNING: High attractor basin density detected!")
        print("  This text may contain areas of significant epistemic confusion.")
    
    if stats.get('total_metaphors', 0) > 10:
        print(f"\n⚠ WARNING: High metaphor density detected!")
        print("  Consider reviewing for areas where more precise language could improve clarity.")
    
    print("="*60)

def main():
    """
    Main CLI entry point for MetaphorScan application.
    
    Implements complete workflow: file validation → text extraction → 
    two-stage analysis → report generation with comprehensive error handling.
    """
    parser = argparse.ArgumentParser(
        description="MetaphorScan: Detect sedative and prophylactic metaphors in AI texts",
        epilog="""
Examples:
  python src/main.py --filepath data/raw/paper.pdf
  python src/main.py --filepath data/raw/article.txt --output reports/analysis.pdf
  python src/main.py --filepath paper.pdf --verbose --chunk-size 2000

Theoretical Framework:
  Based on *The Nuremberg Defense of AI*, *AI as an Epistemic Void Generator*,
  and *The Alignment Problem as Epistemic Autoimmunity* - detecting metaphors
  that shape AI discourse through sedative and prophylactic mechanisms.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--filepath", 
        required=True, 
        help="Path to input file (PDF, TXT, MD)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", 
        default="outputs/reports/metaphorscan_report.pdf",
        help="Path to output PDF report (default: outputs/reports/metaphorscan_report.pdf)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for large file processing (default: auto-detect)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip PDF report generation (analysis only)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = args.log_level
    
    logger = setup_logging(log_level)
    
    try:
        # Print startup banner
        print("\n" + "="*60)
        print("               METAPHORSCAN v1.0")
        print("    Sedative/Prophylactic Metaphor Detection")
        print("="*60)
        
        # Validate input file
        logger.info(f"Validating input file: {args.filepath}")
        input_file = validate_input_file(args.filepath)
        logger.info(f"Input file validated: {input_file}")
        
        # Validate output path
        output_file = validate_output_path(args.output)
        logger.info(f"Output will be saved to: {output_file}")
        
        # Run analysis pipeline
        logger.info("Starting MetaphorScan analysis pipeline")
        print("\nRunning analysis pipeline...")
        print("Stage 1: Text extraction and preprocessing")
        print("Stage 2: Lexical matching with context awareness")  
        print("Stage 3: DistilBERT contextual validation")
        
        # Execute pipeline with optional chunking
        if args.chunk_size:
            logger.info(f"Using chunked processing with chunk size: {args.chunk_size}")
            results = run_pipeline(str(input_file), chunk_size=args.chunk_size)
        else:
            results = run_pipeline(str(input_file))
        
        logger.info("Pipeline analysis completed successfully")
        
        # Print analysis summary
        print_analysis_summary(results, logger)
        
        # Generate PDF report
        if not args.no_report:
            print(f"\nGenerating PDF report: {output_file}")
            logger.info("Starting PDF report generation")
            
            original_text = results.get('original_text', '')
            report_path = generate_metaphor_report(
                results, 
                str(output_file), 
                str(input_file),
                original_text
            )
            
            print(f"✓ PDF report generated: {report_path}")
            logger.info(f"PDF report generated successfully: {report_path}")
        else:
            logger.info("Skipping PDF report generation (--no-report flag)")
        
        # Success message
        print(f"\n✓ MetaphorScan analysis complete!")
        if not args.no_report:
            print(f"  Report saved to: {output_file}")
        print(f"  Log file: outputs/logs/metaphorscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the file path and try again.")
        return 1
        
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        print(f"\n❌ Error: {e}")
        return 1
        
    except MemoryError as e:
        logger.error(f"Memory error during processing: {e}")
        print(f"\n❌ Memory Error: File too large for available memory.")
        print("Try using the --chunk-size option to process in smaller chunks.")
        return 1
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print(f"\n❌ Missing Dependency: {e}")
        print("Please install all required packages from requirements.txt")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        print(f"\n❌ Unexpected Error: {e}")
        print("Please check the log file for detailed error information.")
        return 1

if __name__ == "__main__":
    sys.exit(main())