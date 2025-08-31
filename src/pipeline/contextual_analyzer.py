# MetaphorScan/src/pipeline/contextual_analyzer.py
"""
Contextual analyzer using DistilBERT for semantic metaphor validation.
Implements simplified contextual analysis without fine-tuning, using semantic 
similarity to detect AI-domain metaphors and epistemic attractor basins.

Avoids black-box AI reliance (*Epistemic Autoimmunity*, Section 4) by using
transparent semantic similarity rather than opaque classification.
"""
import os
import logging
import yaml
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(config_path="src/config/settings.yaml"):
    """Load configuration settings."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Settings file not found, using defaults")
        return {
            "pipeline": {"confidence_threshold": 0.7},
            "models": {"transformer_model": "distilbert-base-uncased"}
        }

def load_distilbert_model(model_path=None):
    """
    Load DistilBERT model for semantic analysis.
    
    Uses pre-trained model without fine-tuning to avoid epistemic opacity
    (*The Alignment Problem as Epistemic Autoimmunity*, Section 4).
    """
    if model_path is None:
        settings = load_settings()
        model_name = settings.get("models", {}).get("transformer_model", "distilbert-base-uncased")
        model_path = f"data/models/transformers/{model_name}"
    
    try:
        # Try to load local model first
        if os.path.exists(model_path):
            logger.info(f"Loading local DistilBERT model from {model_path}")
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Fall back to downloading model
            logger.info(f"Local model not found, downloading DistilBERT")
            model_name = "distilbert-base-uncased"
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save for future use
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"Saved model to {model_path}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading DistilBERT model: {e}")
        raise

def get_ai_domain_contexts():
    """
    Define AI domain contexts for semantic similarity comparison.
    
    Based on *Epistemic Void Generator* analysis of how technical terminology
    creates false precision in AI discourse.
    """
    return {
        'technical_contexts': [
            "machine learning algorithm model training",
            "artificial intelligence system processing",
            "neural network computation optimization",
            "deep learning data analysis",
            "AI model prediction generation"
        ],
        'metaphor_contexts': {
            'sedative': [
                "error correction and bug fixing",
                "system failure and malfunction",
                "output mistakes and inaccuracies",
                "performance issues and problems"
            ],
            'prophylactic': [
                "human intelligence and cognition",
                "learning and understanding processes",
                "intelligent behavior and reasoning",
                "training and education methods"
            ]
        }
    }

def get_sentence_embedding(text, model, tokenizer):
    """
    Get contextualized sentence embedding using DistilBERT.
    
    Implements transparent semantic encoding (*Epistemic Autoimmunity*, Section 5)
    without hidden classification layers.
    """
    try:
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use [CLS] token embedding as sentence representation
        sentence_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return sentence_embedding
        
    except Exception as e:
        logger.warning(f"Error getting embedding for text: {e}")
        return None

def calculate_semantic_similarity(text1, text2, model, tokenizer):
    """
    Calculate semantic similarity between two texts using DistilBERT embeddings.
    
    Implements structure-tracking (*Epistemic Autoimmunity*, Section 5) through
    transparent similarity measurement rather than black-box classification.
    """
    emb1 = get_sentence_embedding(text1, model, tokenizer)
    emb2 = get_sentence_embedding(text2, model, tokenizer)
    
    if emb1 is None or emb2 is None:
        return 0.0
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)

def validate_ai_context(sentence, model, tokenizer, threshold=0.6):
    """
    Validate if a sentence occurs in AI-related context using semantic similarity.
    
    Avoids reliance on opaque classification by using transparent similarity
    measurement (*The Alignment Problem as Epistemic Autoimmunity*, Section 4).
    """
    ai_contexts = get_ai_domain_contexts()
    
    max_similarity = 0.0
    best_context = None
    
    # Check similarity to technical AI contexts
    for context in ai_contexts['technical_contexts']:
        similarity = calculate_semantic_similarity(sentence, context, model, tokenizer)
        if similarity > max_similarity:
            max_similarity = similarity
            best_context = context
    
    is_ai_context = max_similarity >= threshold
    
    return {
        'is_ai_context': is_ai_context,
        'similarity_score': max_similarity,
        'best_matching_context': best_context,
        'confidence': max_similarity
    }

def validate_metaphor_context(match, sentence, model, tokenizer):
    """
    Validate metaphor context using semantic similarity to domain-specific patterns.
    
    Implements contextual validation (*Nuremberg Defense*, Section 3) by checking
    whether metaphors appear in their expected epistemic contexts.
    """
    ai_contexts = get_ai_domain_contexts()
    category = match.get('category', 'unknown')
    
    if category not in ai_contexts['metaphor_contexts']:
        return {'is_valid': False, 'similarity_score': 0.0, 'reason': 'unknown_category'}
    
    # Check similarity to category-specific contexts
    category_contexts = ai_contexts['metaphor_contexts'][category]
    max_similarity = 0.0
    best_context = None
    
    for context in category_contexts:
        similarity = calculate_semantic_similarity(sentence, context, model, tokenizer)
        if similarity > max_similarity:
            max_similarity = similarity
            best_context = context
    
    # Also check if it's in AI domain
    ai_validation = validate_ai_context(sentence, model, tokenizer)
    
    # Combine scores - metaphor should be in both AI context AND category context
    combined_score = (max_similarity + ai_validation['similarity_score']) / 2
    
    return {
        'is_valid': combined_score >= 0.5,  # Lower threshold for combined score
        'similarity_score': max_similarity,
        'ai_context_score': ai_validation['similarity_score'],
        'combined_score': combined_score,
        'best_matching_context': best_context,
        'ai_context_match': ai_validation['best_matching_context']
    }

def detect_semantic_attractor_basins(matches, sentences, model, tokenizer):
    """
    Detect epistemic attractor basins using semantic clustering of metaphors.
    
    Implements attractor basin detection (*Epistemic Autoimmunity*, Section 3)
    by identifying semantically coherent clusters of metaphorical language.
    """
    basins = []
    
    # Group sentences with metaphors
    metaphor_sentences = {}
    for match in matches:
        sentence_context = match.get('sentence_context', '')
        if sentence_context not in metaphor_sentences:
            metaphor_sentences[sentence_context] = []
        metaphor_sentences[sentence_context].append(match)
    
    # Analyze semantic coherence of multi-metaphor sentences
    for sentence, sent_matches in metaphor_sentences.items():
        if len(sent_matches) < 2:
            continue
            
        # Calculate semantic similarity between metaphor contexts
        similarities = []
        for i, match1 in enumerate(sent_matches):
            for j, match2 in enumerate(sent_matches[i+1:], i+1):
                term1_context = f"{match1['term']} {match1['category']} {match1['replacement']}"
                term2_context = f"{match2['term']} {match2['category']} {match2['replacement']}"
                
                similarity = calculate_semantic_similarity(term1_context, term2_context, model, tokenizer)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            
            # High semantic similarity suggests coordinated metaphorical strategy
            if avg_similarity > 0.4:  # Threshold for semantic basin
                basin = {
                    'type': 'semantic_basin',
                    'sentence': sentence,
                    'metaphor_count': len(sent_matches),
                    'metaphors': sent_matches,
                    'semantic_coherence': avg_similarity,
                    'categories': list(set(m['category'] for m in sent_matches)),
                    'basin_strength': avg_similarity * len(sent_matches),
                    'description': f"Semantic attractor basin with {len(sent_matches)} metaphors "
                                 f"(coherence: {avg_similarity:.3f})"
                }
                
                basins.append(basin)
                logger.info(f"Detected semantic basin: {len(sent_matches)} metaphors, "
                           f"coherence: {avg_similarity:.3f}")
    
    return basins

def contextual_analyze(text: str, lexical_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main contextual analysis function using DistilBERT for metaphor validation.
    
    Args:
        text: Input text
        lexical_results: Results from lexical matching stage
        
    Returns:
        Dict containing validated matches and contextual analysis
        
    Implements simplified contextual analysis (*Epistemic Autoimmunity*, Section 4)
    using semantic similarity rather than opaque neural classification.
    """
    # Load model and settings
    model, tokenizer = load_distilbert_model()
    settings = load_settings()
    confidence_threshold = settings.get("pipeline", {}).get("confidence_threshold", 0.7)
    
    # Extract lexical matches
    if isinstance(lexical_results, list):
        # Handle legacy format
        lexical_matches = lexical_results
        sentences = []
    else:
        # Handle new enhanced format
        lexical_matches = lexical_results.get('matches', [])
        sentences = lexical_results.get('preprocessed_data', {}).get('sentences', [])
    
    validated_matches = []
    validation_details = []
    
    logger.info(f"Validating {len(lexical_matches)} lexical matches with DistilBERT")
    
    # Validate each lexical match
    for match in lexical_matches:
        sentence_context = match.get('sentence_context', text)
        
        # Validate AI context
        ai_validation = validate_ai_context(sentence_context, model, tokenizer)
        
        # Validate metaphor-specific context
        metaphor_validation = validate_metaphor_context(match, sentence_context, model, tokenizer)
        
        # Calculate combined confidence
        lexical_confidence = match.get('confidence', 0.5)
        context_confidence = ai_validation['confidence']
        metaphor_confidence = metaphor_validation.get('combined_score', 0.0)
        
        # Weighted combination of confidences
        final_confidence = (
            0.4 * lexical_confidence +      # Lexical match weight
            0.3 * context_confidence +      # AI context weight  
            0.3 * metaphor_confidence       # Metaphor context weight
        )
        
        # Validation decision
        is_validated = (
            final_confidence >= confidence_threshold and
            ai_validation['is_ai_context'] and
            metaphor_validation['is_valid']
        )
        
        validation_detail = {
            'original_match': match,
            'ai_validation': ai_validation,
            'metaphor_validation': metaphor_validation,
            'final_confidence': final_confidence,
            'is_validated': is_validated,
            'validation_factors': {
                'lexical_confidence': lexical_confidence,
                'ai_context_confidence': context_confidence,
                'metaphor_context_confidence': metaphor_confidence
            }
        }
        
        validation_details.append(validation_detail)
        
        if is_validated:
            validated_match = match.copy()
            validated_match['confidence'] = final_confidence
            validated_match['validation_details'] = {
                'ai_context_similarity': context_confidence,
                'metaphor_context_similarity': metaphor_confidence,
                'validation_method': 'distilbert_semantic_similarity'
            }
            validated_matches.append(validated_match)
            
            logger.debug(f"Validated metaphor: {match['term']} ({match['category']}) "
                        f"with confidence {final_confidence:.3f}")
    
    # Detect semantic attractor basins
    semantic_basins = detect_semantic_attractor_basins(validated_matches, sentences, model, tokenizer)
    
    # Calculate statistics
    total_validated = len(validated_matches)
    validation_rate = total_validated / max(len(lexical_matches), 1)
    sedative_validated = len([m for m in validated_matches if m['category'] == 'sedative'])
    prophylactic_validated = len([m for m in validated_matches if m['category'] == 'prophylactic'])
    
    logger.info(f"Contextual validation complete: {total_validated}/{len(lexical_matches)} "
                f"matches validated ({validation_rate:.2%})")
    logger.info(f"Semantic basins detected: {len(semantic_basins)}")
    
    return {
        'validated_matches': validated_matches,
        'semantic_basins': semantic_basins,
        'validation_details': validation_details,
        'statistics': {
            'total_lexical_matches': len(lexical_matches),
            'total_validated': total_validated,
            'validation_rate': validation_rate,
            'sedative_validated': sedative_validated,
            'prophylactic_validated': prophylactic_validated,
            'semantic_basin_count': len(semantic_basins),
            'average_confidence': np.mean([m['confidence'] for m in validated_matches]) if validated_matches else 0.0
        },
        'model_info': {
            'model_type': 'distilbert-base-uncased',
            'validation_method': 'semantic_similarity',
            'confidence_threshold': confidence_threshold
        }
    }

if __name__ == "__main__":
    # Test contextual analysis
    sample_matches = [
        {
            "term": "hallucination", 
            "category": "sedative", 
            "replacement": "fabrication", 
            "confidence": 0.8,
            "sentence_context": "The AI model hallucinated incorrect outputs during inference."
        },
        {
            "term": "intelligence", 
            "category": "prophylactic", 
            "replacement": "pattern simulation", 
            "confidence": 0.75,
            "sentence_context": "The system showed remarkable artificial intelligence capabilities."
        },
        {
            "term": "training", 
            "category": "prophylactic", 
            "replacement": "data conditioning", 
            "confidence": 0.7,
            "sentence_context": "The model requires extensive training on large datasets."
        }
    ]
    
    sample_text = """
    The artificial intelligence model demonstrated remarkable learning capabilities during training.
    However, the system occasionally hallucinated outputs that diverged from expected patterns.
    This suggests alignment challenges in human-AI interaction paradigms.
    """
    
    try:
        # Test with legacy format
        results = contextual_analyze(sample_text, sample_matches)
        
        print("Contextual Analysis Results:")
        print(f"Validated matches: {results['statistics']['total_validated']}")
        print(f"Validation rate: {results['statistics']['validation_rate']:.2%}")
        print(f"Semantic basins: {results['statistics']['semantic_basin_count']}")
        print(f"Average confidence: {results['statistics']['average_confidence']:.3f}")
        
        print("\nValidated Metaphors:")
        for match in results['validated_matches']:
            print(f"  - '{match['term']}' ({match['category']}) -> '{match['replacement']}' "
                  f"[confidence: {match['confidence']:.3f}]")
        
        print("\nSemantic Basins:")
        for basin in results['semantic_basins']:
            print(f"  - {basin['description']}")
            
    except Exception as e:
        print(f"Error in contextual analysis: {e}")
        import traceback
        traceback.print_exc()