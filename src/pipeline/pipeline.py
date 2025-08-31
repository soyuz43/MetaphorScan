# MetaphorScan/src/pipeline/pipeline.py
"""
Integrated MetaphorScan pipeline orchestrating the two-step analysis process.
Coordinates text extraction, preprocessing, lexical matching, and contextual validation
to detect sedative/prophylactic metaphors and epistemic attractor basins.

Implements the complete analysis framework from *The Nuremberg Defense of AI*,
*AI as an Epistemic Void Generator*, and *The Alignment Problem as Epistemic Autoimmunity*.
"""
import logging
import yaml
from typing import Dict, Any, Optional
from text_processing.extract import extract_text, extract_text_chunks
from text_processing.preprocess import preprocess_text, preprocess_for_chunked_analysis
from pipeline.lexical_matcher import lexical_match
from pipeline.contextual_analyzer import contextual_analyze

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(config_path="src/config/settings.yaml"):
    """Load pipeline configuration settings."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Settings file not found, using defaults")
        return {
            "pipeline": {
                "confidence_threshold": 0.7,
                "max_file_size_mb": 50
            }
        }

def run_pipeline(filepath: str, chunk_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute the complete MetaphorScan analysis pipeline.
    
    Implements two-stage detection (*Nuremberg Defense*, *Epistemic Autoimmunity*):
    1. Lexical matching with context-aware scoring 
    2. DistilBERT contextual validation using semantic similarity
    
    Args:
        filepath: Path to input file (PDF or text)
        chunk_size: Optional chunk size for large file processing
        
    Returns:
        Dict containing comprehensive analysis results
    """
    logger.info(f"Starting MetaphorScan pipeline for: {filepath}")
    
    try:
        # Load configuration
        settings = load_settings()
        confidence_threshold = settings.get("pipeline", {}).get("confidence_threshold", 0.7)
        max_file_size = settings.get("pipeline", {}).get("max_file_size_mb", 50)
        
        # Stage 0: Text extraction
        logger.info("Stage 0: Extracting text from input file")
        if chunk_size:
            logger.info(f"Using chunked processing with chunk size: {chunk_size}")
            return run_chunked_pipeline(filepath, chunk_size, settings)
        else:
            original_text = extract_text(filepath)
            logger.info(f"Extracted {len(original_text)} characters")
        
        # Stage 0.5: Text preprocessing  
        logger.info("Stage 0.5: Preprocessing text")
        preprocessed_data = preprocess_text(original_text)
        logger.info(f"Preprocessed into {preprocessed_data['statistics']['sentence_count']} sentences "
                   f"with {preprocessed_data['statistics']['ai_term_count']} AI terms")
        
        # Stage 1: Lexical matching with context awareness
        logger.info("Stage 1: Lexical matching with context-aware scoring")
        lexical_results = lexical_match(original_text, preprocessed_data)
        lexical_matches = lexical_results['matches']
        lexical_basins = lexical_results['attractor_basins']
        
        logger.info(f"Stage 1 complete: {len(lexical_matches)} metaphors found, "
                   f"{len(lexical_basins)} lexical attractor basins")
        
        # Stage 2: Contextual validation with DistilBERT
        logger.info("Stage 2: Contextual validation with DistilBERT semantic analysis")
        contextual_results = contextual_analyze(original_text, lexical_results)
        validated_matches = contextual_results['validated_matches']
        semantic_basins = contextual_results.get('semantic_basins', [])
        
        logger.info(f"Stage 2 complete: {len(validated_matches)} metaphors validated, "
                   f"{len(semantic_basins)} semantic basins detected")
        
        # Combine results for comprehensive analysis
        all_attractor_basins = lexical_basins + semantic_basins
        
        # Calculate final statistics
        final_statistics = {
            'total_metaphors': len(validated_matches),
            'sedative_count': len([m for m in validated_matches if m['category'] == 'sedative']),
            'prophylactic_count': len([m for m in validated_matches if m['category'] == 'prophylactic']),
            'lexical_detection_count': len(lexical_matches),
            'validation_rate': len(validated_matches) / max(len(lexical_matches), 1),
            'attractor_basin_count': len(all_attractor_basins),
            'lexical_basin_count': len(lexical_basins),
            'semantic_basin_count': len(semantic_basins),
            'average_confidence': sum(m['confidence'] for m in validated_matches) / max(len(validated_matches), 1),
            'text_length': len(original_text),
            'sentence_count': preprocessed_data['statistics']['sentence_count'],
            'metaphor_density': len(validated_matches) / max(preprocessed_data['statistics']['sentence_count'], 1),
            'pipeline_settings': {
                'confidence_threshold': confidence_threshold,
                'max_file_size_mb': max_file_size
            }
        }
        
        # Compile comprehensive results
        pipeline_results = {
            'validated_matches': validated_matches,
            'lexical_results': lexical_results,
            'contextual_results': contextual_results,
            'attractor_basins': lexical_basins,  # Legacy format compatibility
            'semantic_basins': semantic_basins,
            'all_attractor_basins': all_attractor_basins,
            'statistics': final_statistics,
            'preprocessed_data': preprocessed_data,
            'original_text': original_text,
            'metadata': {
                'input_file': filepath,
                'analysis_method': 'two_stage_pipeline',
                'lexical_method': 'spacy_context_aware',
                'contextual_method': 'distilbert_semantic_similarity',
                'theoretical_framework': [
                    'The Nuremberg Defense of AI',
                    'AI as an Epistemic Void Generator', 
                    'The Alignment Problem as Epistemic Autoimmunity'
                ]
            }
        }
        
        logger.info(f"Pipeline complete: {final_statistics['total_metaphors']} metaphors validated "
                   f"({final_statistics['validation_rate']:.1%} validation rate), "
                   f"{final_statistics['attractor_basin_count']} attractor basins detected")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed for {filepath}: {e}")
        raise

def run_chunked_pipeline(filepath: str, chunk_size: int, settings: Dict) -> Dict[str, Any]:
    """
    Run pipeline on large files using chunked processing.
    
    Implements memory-efficient analysis for large documents while maintaining
    context across chunks (*Epistemic Autoimmunity*, Section 4).
    """
    logger.info(f"Running chunked pipeline with chunk size: {chunk_size}")
    
    # Extract text in chunks
    text_chunks = list(extract_text_chunks(filepath, chunk_size))
    logger.info(f"Split into {len(text_chunks)} chunks")
    
    # Preprocess chunks
    preprocessed_chunks = preprocess_for_chunked_analysis(
        "\n\n".join(text_chunks), chunk_size
    )
    
    # Analyze each chunk
    all_validated_matches = []
    all_lexical_basins = []
    all_semantic_basins = []
    chunk_statistics = []
    
    for i, chunk_data in enumerate(preprocessed_chunks):
        logger.info(f"Processing chunk {i+1}/{len(preprocessed_chunks)}")
        
        chunk_text = chunk_data['text']
        
        # Lexical analysis for chunk
        lexical_results = lexical_match(chunk_text, {'sentences': [], 'ai_terms': chunk_data['ai_terms']})
        
        # Contextual analysis for chunk
        contextual_results = contextual_analyze(chunk_text, lexical_results)
        
        # Accumulate results
        all_validated_matches.extend(contextual_results['validated_matches'])
        all_lexical_basins.extend(lexical_results.get('attractor_basins', []))
        all_semantic_basins.extend(contextual_results.get('semantic_basins', []))
        
        chunk_stats = {
            'chunk_id': i,
            'text_length': len(chunk_text),
            'metaphors_found': len(contextual_results['validated_matches']),
            'basins_found': len(lexical_results.get('attractor_basins', [])) + len(contextual_results.get('semantic_basins', []))
        }
        chunk_statistics.append(chunk_stats)
    
    # Combine all text for full context
    full_text = "\n\n".join(text_chunks)
    
    # Calculate aggregate statistics
    final_statistics = {
        'total_metaphors': len(all_validated_matches),
        'sedative_count': len([m for m in all_validated_matches if m['category'] == 'sedative']),
        'prophylactic_count': len([m for m in all_validated_matches if m['category'] == 'prophylactic']),
        'attractor_basin_count': len(all_lexical_basins) + len(all_semantic_basins),
        'lexical_basin_count': len(all_lexical_basins),
        'semantic_basin_count': len(all_semantic_basins),
        'average_confidence': sum(m['confidence'] for m in all_validated_matches) / max(len(all_validated_matches), 1),
        'text_length': len(full_text),
        'chunk_count': len(text_chunks),
        'chunk_statistics': chunk_statistics,
        'processing_method': 'chunked_analysis'
    }
    
    return {
        'validated_matches': all_validated_matches,
        'attractor_basins': all_lexical_basins,
        'semantic_basins': all_semantic_basins,
        'all_attractor_basins': all_lexical_basins + all_semantic_basins,
        'statistics': final_statistics,
        'original_text': full_text,
        'chunk_data': {
            'chunk_size': chunk_size,
            'chunk_count': len(text_chunks),
            'chunk_statistics': chunk_statistics
        },
        'metadata': {
            'input_file': filepath,
            'analysis_method': 'chunked_two_stage_pipeline',
            'chunk_size': chunk_size
        }
    }

# Test and development functions
if __name__ == "__main__":
    # Test pipeline with sample data
    import os
    
    # Create sample test file
    sample_text = """
    The artificial intelligence model demonstrated remarkable learning capabilities during training.
    However, the system occasionally hallucinated outputs that diverged from expected patterns.
    This error rate suggests alignment challenges in human-AI interaction paradigms.
    The model's intelligence seems to emerge from complex training procedures.
    """
    
    sample_file = "data/raw/sample.txt"
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    try:
        # Test standard pipeline
        logger.info("Testing standard pipeline")
        results = run_pipeline(sample_file)
        
        print("\n=== Pipeline Test Results ===")
        print(f"Total metaphors: {results['statistics']['total_metaphors']}")
        print(f"Sedative: {results['statistics']['sedative_count']}")
        print(f"Prophylactic: {results['statistics']['prophylactic_count']}")
        print(f"Validation rate: {results['statistics']['validation_rate']:.1%}")
        print(f"Attractor basins: {results['statistics']['attractor_basin_count']}")
        print(f"Average confidence: {results['statistics']['average_confidence']:.3f}")
        
        print("\nDetected Metaphors:")
        for match in results['validated_matches']:
            print(f"  - '{match['term']}' ({match['category']}) -> '{match['replacement']}' "
                  f"[{match['confidence']:.3f}]")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()