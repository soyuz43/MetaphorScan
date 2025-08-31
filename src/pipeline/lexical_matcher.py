# MetaphorScan/src/pipeline/lexical_matcher.py
"""
Enhanced lexical matcher with context-aware metaphor detection.
Uses spaCy's dependency parsing for proximity-based scoring near AI terms.

Implements rule-based detection following *The Nuremberg Defense of AI* analysis
of how sedative/prophylactic metaphors operate in technical discourse.
"""
import spacy
import yaml
import os
import logging
from typing import List, Dict, Any, Optional
from text_processing.preprocess import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_lexicon(config_path="src/config/lexicon.yaml"):
    """Load metaphor lexicon from YAML configuration."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Lexicon file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing lexicon YAML: {e}")
        raise

def load_settings(config_path="src/config/settings.yaml"):
    """Load pipeline settings."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Settings file not found, using defaults")
        return {"pipeline": {"confidence_threshold": 0.7}}

def get_ai_context_terms():
    """
    Define AI-related terms that provide context for metaphor detection.
    
    Based on *Epistemic Void Generator* analysis of technical terminology
    that creates epistemic authority through apparent precision.
    """
    return {
        'technical_terms': {
            'model', 'algorithm', 'neural', 'network', 'system', 'ai', 'ml',
            'machine', 'artificial', 'deep', 'learning', 'training', 'data'
        },
        'process_terms': {
            'output', 'input', 'generation', 'prediction', 'classification',
            'optimization', 'processing', 'computation', 'inference'
        },
        'performance_terms': {
            'accuracy', 'performance', 'efficiency', 'robustness', 'reliability',
            'evaluation', 'metrics', 'benchmark', 'validation'
        },
        'safety_terms': {
            'alignment', 'safety', 'control', 'governance', 'risk', 'bias',
            'fairness', 'transparency', 'explainability', 'interpretability'
        }
    }

def calculate_context_score(token, doc, ai_terms, window_size=10):
    """
    Calculate contextual relevance score based on proximity to AI terms.
    
    Implements proximity-based scoring (*Nuremberg Defense*, Section 3) where
    metaphors gain significance through association with technical authority.
    
    Args:
        token: spaCy token containing potential metaphor
        doc: spaCy document object
        ai_terms: Set of AI-related terms
        window_size: Word window for context analysis
        
    Returns:
        float: Context score (0.0 to 1.0)
    """
    context_score = 0.0
    token_index = token.i
    
    # Check words within window
    start_idx = max(0, token_index - window_size)
    end_idx = min(len(doc), token_index + window_size + 1)
    
    ai_term_count = 0
    total_terms = 0
    
    for i in range(start_idx, end_idx):
        if i == token_index:
            continue
            
        neighbor_token = doc[i]
        total_terms += 1
        
        # Check if neighbor is AI-related term
        neighbor_text = neighbor_token.text.lower()
        neighbor_lemma = neighbor_token.lemma_.lower()
        
        for category_terms in ai_terms.values():
            if neighbor_text in category_terms or neighbor_lemma in category_terms:
                ai_term_count += 1
                break
    
    if total_terms > 0:
        context_score = ai_term_count / total_terms
    
    # Boost score for direct syntactic relationships
    dependency_boost = 0.0
    if token.head.text.lower() in set().union(*ai_terms.values()):
        dependency_boost += 0.3
    
    for child in token.children:
        if child.text.lower() in set().union(*ai_terms.values()):
            dependency_boost += 0.2
    
    final_score = min(1.0, context_score + dependency_boost)
    return final_score

def analyze_metaphor_strength(token, lexicon_item, context_score):
    """
    Analyze the strength of a metaphor match based on linguistic features.
    
    Implements metaphor strength assessment (*Epistemic Autoimmunity*, Section 2)
    considering how metaphors mask structural issues through linguistic patterns.
    
    Args:
        token: spaCy token
        lexicon_item: Lexicon entry for the metaphor
        context_score: Contextual relevance score
        
    Returns:
        float: Metaphor strength confidence (0.0 to 1.0)
    """
    base_confidence = 0.6  # Base confidence for exact match
    
    # Boost confidence based on context
    context_boost = context_score * 0.3
    
    # Consider part-of-speech for metaphor likelihood
    pos_boost = 0.0
    if token.pos_ in ['NOUN', 'VERB']:  # Common metaphor carriers
        pos_boost = 0.1
    elif token.pos_ in ['ADJ']:  # Adjectives can be metaphorical
        pos_boost = 0.05
    
    # Consider morphological variants
    morph_boost = 0.0
    if token.text.lower() != token.lemma_.lower():  # Inflected form
        morph_boost = 0.05
    
    # Consider frequency in technical contexts (simple heuristic)
    freq_adjustment = 0.0
    if len(token.text) > 8:  # Longer technical terms
        freq_adjustment = 0.05
    
    final_confidence = min(1.0, base_confidence + context_boost + pos_boost + morph_boost + freq_adjustment)
    return final_confidence

def find_sentence_metaphors(sentence_data, lexicon, ai_terms):
    """
    Find metaphors within a sentence with enhanced context analysis.
    
    Sentence-level analysis enables detection of epistemic attractor basins
    (*Epistemic Autoimmunity*, Section 3) where multiple metaphors cluster.
    """
    matches = []
    sentence_text = sentence_data['text']
    
    # Re-process sentence with spaCy for detailed analysis
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence_text)
    
    for token in doc:
        token_lower = token.text.lower()
        token_lemma = token.lemma_.lower()
        
        # Check against lexicon
        for category in ["sedative", "prophylactic"]:
            for item in lexicon[category]:
                term_lower = item["term"].lower()
                
                # Match exact term or lemmatized form
                if token_lower == term_lower or token_lemma == term_lower:
                    # Calculate context score
                    context_score = calculate_context_score(token, doc, ai_terms)
                    
                    # Analyze metaphor strength
                    confidence = analyze_metaphor_strength(token, item, context_score)
                    
                    match = {
                        "term": token.text,
                        "lemma": token.lemma_,
                        "category": category,
                        "replacement": item["replacement"],
                        "description": item["description"],
                        "confidence": confidence,
                        "context_score": context_score,
                        "position": {
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "sentence_start": sentence_data.get('start', 0),
                            "sentence_end": sentence_data.get('end', len(sentence_text))
                        },
                        "linguistic_features": {
                            "pos": token.pos_,
                            "dep": token.dep_,
                            "head": token.head.text,
                            "children": [child.text for child in token.children]
                        },
                        "sentence_context": sentence_text
                    }
                    
                    matches.append(match)
                    logger.debug(f"Found metaphor: {token.text} ({category}) with confidence {confidence:.3f}")
    
    return matches

def detect_attractor_basins(matches, sentences):
    """
    Detect epistemic attractor basins - areas with high metaphor density.
    
    Implements attractor basin detection (*Epistemic Autoimmunity*, Section 3)
    where multiple metaphors cluster to create zones of epistemic confusion.
    
    Args:
        matches: List of metaphor matches
        sentences: List of sentence data
        
    Returns:
        List[Dict]: Detected attractor basins
    """
    basins = []
    
    # Group matches by sentence
    sentence_metaphors = {}
    for match in matches:
        sent_start = match['position']['sentence_start']
        if sent_start not in sentence_metaphors:
            sentence_metaphors[sent_start] = []
        sentence_metaphors[sent_start].append(match)
    
    # Look for high-density areas
    for sent_start, sent_matches in sentence_metaphors.items():
        if len(sent_matches) >= 2:  # Multiple metaphors in same sentence
            # Find the sentence text
            sentence_text = next((s['text'] for s in sentences if s.get('start') == sent_start), "")
            
            basin = {
                'sentence': sentence_text,
                'metaphor_count': len(sent_matches),
                'metaphors': sent_matches,
                'density_score': len(sent_matches) / max(len(sentence_text.split()), 1),
                'categories': list(set(m['category'] for m in sent_matches)),
                'avg_confidence': sum(m['confidence'] for m in sent_matches) / len(sent_matches),
                'position': {
                    'start': sent_start,
                    'end': sent_matches[0]['position']['sentence_end']
                }
            }
            
            basins.append(basin)
            logger.info(f"Detected attractor basin with {len(sent_matches)} metaphors: {sentence_text[:100]}...")
    
    return basins

def lexical_match(text: str, preprocessed_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced lexical matching with context-aware metaphor detection.
    
    Args:
        text: Input text for analysis
        preprocessed_data: Optional preprocessed text data (for efficiency)
        
    Returns:
        Dict containing matches, attractor basins, and analysis metadata
        
    Implements enhanced lexical analysis (*Nuremberg Defense*, Section 3) with
    proximity-based scoring and attractor basin detection.
    """
    # Load configuration
    lexicon = load_lexicon()
    ai_terms = get_ai_context_terms()
    
    # Use preprocessed data if available, otherwise preprocess
    if preprocessed_data is None:
        preprocessed_data = preprocess_text(text)
    
    sentences = preprocessed_data['sentences']
    matches = []
    
    # Process each sentence for metaphors
    for sentence_data in sentences:
        sentence_matches = find_sentence_metaphors(sentence_data, lexicon, ai_terms)
        matches.extend(sentence_matches)
    
    # Detect epistemic attractor basins
    attractor_basins = detect_attractor_basins(matches, sentences)
    
    # Calculate overall statistics
    total_metaphors = len(matches)
    sedative_count = len([m for m in matches if m['category'] == 'sedative'])
    prophylactic_count = len([m for m in matches if m['category'] == 'prophylactic'])
    avg_confidence = sum(m['confidence'] for m in matches) / max(total_metaphors, 1)
    
    logger.info(f"Lexical analysis complete: {total_metaphors} metaphors found "
                f"({sedative_count} sedative, {prophylactic_count} prophylactic)")
    logger.info(f"Average confidence: {avg_confidence:.3f}, Attractor basins: {len(attractor_basins)}")
    
    return {
        'matches': matches,
        'attractor_basins': attractor_basins,
        'statistics': {
            'total_metaphors': total_metaphors,
            'sedative_count': sedative_count,
            'prophylactic_count': prophylactic_count,
            'average_confidence': avg_confidence,
            'attractor_basin_count': len(attractor_basins),
            'sentence_count': len(sentences),
            'metaphor_density': total_metaphors / max(len(sentences), 1)
        },
        'preprocessed_data': preprocessed_data
    }

if __name__ == "__main__":
    # Test enhanced lexical matching
    sample_text = """
    The artificial intelligence model demonstrated remarkable learning capabilities during training.
    However, the system occasionally hallucinated outputs that diverged from expected patterns.
    This error rate suggests alignment challenges in human-AI interaction paradigms.
    The model's intelligence seems to emerge from complex training procedures.
    """
    
    try:
        results = lexical_match(sample_text)
        
        print("Enhanced Lexical Analysis Results:")
        print(f"Total metaphors: {results['statistics']['total_metaphors']}")
        print(f"Sedative metaphors: {results['statistics']['sedative_count']}")
        print(f"Prophylactic metaphors: {results['statistics']['prophylactic_count']}")
        print(f"Average confidence: {results['statistics']['average_confidence']:.3f}")
        print(f"Attractor basins: {results['statistics']['attractor_basin_count']}")
        
        print("\nDetected Metaphors:")
        for match in results['matches']:
            print(f"  - '{match['term']}' ({match['category']}) -> '{match['replacement']}' "
                  f"[confidence: {match['confidence']:.3f}, context: {match['context_score']:.3f}]")
        
        print("\nAttractor Basins:")
        for basin in results['attractor_basins']:
            print(f"  - {basin['metaphor_count']} metaphors in: {basin['sentence'][:100]}...")
            
    except Exception as e:
        print(f"Error in lexical matching: {e}")
        import traceback
        traceback.print_exc()