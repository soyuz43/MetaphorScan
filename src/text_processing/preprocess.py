# MetaphorScan/src/text_processing/preprocess.py
"""
Text preprocessing module using spaCy for pipeline input preparation.
Implements minimal preprocessing to preserve metaphorical language patterns.

Avoids over-normalization that could obscure sedative/prophylactic metaphors,
following *The Alignment Problem as Epistemic Autoimmunity* principle of 
maintaining linguistic authenticity.
"""
import spacy
import re
import logging
from typing import List, Dict, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(config_path="src/config/settings.yaml"):
    """Load spaCy model configuration from settings."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
            return settings.get("models", {}).get("spacy_model", "en_core_web_sm")
    except FileNotFoundError:
        logger.warning(f"Settings file not found, using default spaCy model")
        return "en_core_web_sm"

def load_spacy_model(model_name=None):
    """
    Load spaCy model with error handling.
    
    Implements structure-tracking (*Epistemic Autoimmunity*, Section 5) by
    maintaining linguistic structure through dependency parsing.
    """
    if model_name is None:
        model_name = load_settings()
    
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        logger.error(f"spaCy model '{model_name}' not found. Please install with: python -m spacy download {model_name}")
        raise
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Basic text cleaning while preserving metaphorical content.
    
    Minimal cleaning approach to avoid destroying sedative/prophylactic metaphors
    that may appear in various linguistic forms (*Nuremberg Defense*, Section 1).
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces/tabs
    text = text.strip()
    
    # Remove common PDF artifacts without destroying content
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    text = re.sub(r'­', '', text)    # Soft hyphens
    
    # Preserve hyphenated terms (important for AI terminology)
    # Don't split "AI-related", "human-like", etc.
    
    return text

def preserve_ai_terminology(doc):
    """
    Identify and preserve AI-related terminology during preprocessing.
    
    Critical for contextual analysis - maintains AI domain terms that provide
    context for metaphor detection (*Epistemic Void Generator*, Section 2).
    
    Args:
        doc: spaCy document object
        
    Returns:
        List[Dict]: AI-related terms with positions
    """
    ai_terms = {
        'model', 'algorithm', 'neural', 'network', 'training', 'learning',
        'intelligence', 'artificial', 'machine', 'deep', 'ai', 'ml',
        'data', 'dataset', 'output', 'input', 'prediction', 'classification',
        'regression', 'optimization', 'gradient', 'backpropagation',
        'transformer', 'attention', 'embedding', 'token', 'llm',
        'alignment', 'safety', 'robustness', 'hallucination', 'bias'
    }
    
    ai_entities = []
    for token in doc:
        if token.text.lower() in ai_terms:
            ai_entities.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'start': token.idx,
                'end': token.idx + len(token.text),
                'is_ai_term': True
            })
    
    return ai_entities

def extract_sentences(doc) -> List[Dict[str, Any]]:
    """
    Extract sentences with metadata for metaphor analysis.
    
    Sentence-level analysis enables detection of epistemic attractor basins
    (*Epistemic Autoimmunity*, Section 3) by identifying metaphor clustering.
    
    Args:
        doc: spaCy document object
        
    Returns:
        List[Dict]: Sentences with metadata
    """
    sentences = []
    
    for sent in doc.sents:
        # Skip very short sentences that unlikely contain metaphors
        if len(sent.text.strip()) < 10:
            continue
            
        sentence_data = {
            'text': sent.text.strip(),
            'start': sent.start_char,
            'end': sent.end_char,
            'tokens': [token.text for token in sent],
            'lemmas': [token.lemma_ for token in sent],
            'pos_tags': [token.pos_ for token in sent],
            'dependencies': [(token.text, token.dep_, token.head.text) for token in sent],
            'entities': [(ent.text, ent.label_) for ent in sent.ents],
            'length': len(sent.text)
        }
        
        sentences.append(sentence_data)
    
    return sentences

def identify_metaphor_contexts(doc) -> List[Dict[str, Any]]:
    """
    Identify linguistic contexts that commonly contain metaphors.
    
    Implements pattern recognition for sedative/prophylactic metaphor detection,
    based on linguistic analysis from *The Nuremberg Defense of AI*.
    
    Args:
        doc: spaCy document object
        
    Returns:
        List[Dict]: Potential metaphor contexts
    """
    contexts = []
    
    # Look for specific patterns that often contain metaphors
    metaphor_patterns = [
        # Anthropomorphic patterns (prophylactic metaphors)
        {'pattern': 'intelligence', 'context_words': ['artificial', 'machine', 'human-like']},
        {'pattern': 'learning', 'context_words': ['deep', 'machine', 'supervised']},
        {'pattern': 'training', 'context_words': ['model', 'data', 'algorithm']},
        
        # Error minimization patterns (sedative metaphors)  
        {'pattern': 'hallucination', 'context_words': ['model', 'output', 'generation']},
        {'pattern': 'error', 'context_words': ['rate', 'correction', 'minimization']},
        {'pattern': 'alignment', 'context_words': ['value', 'human', 'goal']},
    ]
    
    for sent in doc.sents:
        sent_text_lower = sent.text.lower()
        
        for pattern in metaphor_patterns:
            if pattern['pattern'] in sent_text_lower:
                # Check for context words
                context_score = sum(1 for word in pattern['context_words'] 
                                  if word in sent_text_lower)
                
                if context_score > 0:
                    contexts.append({
                        'sentence': sent.text,
                        'pattern': pattern['pattern'],
                        'context_score': context_score,
                        'start': sent.start_char,
                        'end': sent.end_char,
                        'context_words_found': [word for word in pattern['context_words'] 
                                              if word in sent_text_lower]
                    })
    
    return contexts

def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Main preprocessing function that prepares text for metaphor detection pipeline.
    
    Returns structured data preserving linguistic information needed for both
    lexical matching and contextual analysis steps.
    
    Args:
        text (str): Raw input text
        
    Returns:
        Dict: Preprocessed text with metadata
        
    Implements minimal preprocessing approach (*Epistemic Autoimmunity*, Section 5)
    to preserve metaphorical language while enabling systematic analysis.
    """
    # Clean text minimally
    cleaned_text = clean_text(text)
    
    # Load spaCy model and process
    nlp = load_spacy_model()
    doc = nlp(cleaned_text)
    
    # Extract structured information
    sentences = extract_sentences(doc)
    ai_terms = preserve_ai_terminology(doc)
    metaphor_contexts = identify_metaphor_contexts(doc)
    
    # Calculate basic statistics
    word_count = len([token for token in doc if not token.is_space])
    sentence_count = len(list(doc.sents))
    
    logger.info(f"Preprocessed text: {word_count} words, {sentence_count} sentences")
    logger.info(f"Found {len(ai_terms)} AI-related terms, {len(metaphor_contexts)} potential metaphor contexts")
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'doc': doc,  # spaCy document for further analysis
        'sentences': sentences,
        'ai_terms': ai_terms,
        'metaphor_contexts': metaphor_contexts,
        'statistics': {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'ai_term_count': len(ai_terms),
            'metaphor_context_count': len(metaphor_contexts)
        }
    }

def preprocess_for_chunked_analysis(text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Preprocess text in chunks for memory-efficient analysis of large documents.
    
    Maintains sentence boundaries and context across chunks to preserve
    metaphorical meaning (*Nuremberg Defense*, Section 2).
    
    Args:
        text (str): Input text
        chunk_size (int): Target chunk size in characters
        
    Returns:
        List[Dict]: Preprocessed chunks with metadata
    """
    chunks = []
    nlp = load_spacy_model()
    
    # Split into chunks while preserving sentences
    doc = nlp(text)
    current_chunk = ""
    current_sentences = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        if len(current_chunk) + len(sent_text) <= chunk_size:
            current_chunk += sent_text + " "
            current_sentences.append(sent)
        else:
            # Process current chunk
            if current_chunk.strip():
                chunk_doc = nlp(current_chunk.strip())
                chunks.append({
                    'text': current_chunk.strip(),
                    'doc': chunk_doc,
                    'sentence_count': len(current_sentences),
                    'ai_terms': preserve_ai_terminology(chunk_doc),
                    'metaphor_contexts': identify_metaphor_contexts(chunk_doc)
                })
            
            # Start new chunk
            current_chunk = sent_text + " "
            current_sentences = [sent]
    
    # Process final chunk
    if current_chunk.strip():
        chunk_doc = nlp(current_chunk.strip())
        chunks.append({
            'text': current_chunk.strip(),
            'doc': chunk_doc,
            'sentence_count': len(current_sentences),
            'ai_terms': preserve_ai_terminology(chunk_doc),
            'metaphor_contexts': identify_metaphor_contexts(chunk_doc)
        })
    
    logger.info(f"Split text into {len(chunks)} chunks for analysis")
    return chunks

# Test function for development
if __name__ == "__main__":
    # Test preprocessing with sample AI text
    sample_text = """
    The artificial intelligence model demonstrated remarkable learning capabilities during training.
    However, the system occasionally hallucinated outputs that diverged from expected patterns.
    This error rate suggests alignment challenges in human-AI interaction paradigms.
    """
    
    try:
        # Test standard preprocessing
        result = preprocess_text(sample_text)
        print("Preprocessing Results:")
        print(f"Word count: {result['statistics']['word_count']}")
        print(f"Sentences: {result['statistics']['sentence_count']}")
        print(f"AI terms found: {len(result['ai_terms'])}")
        print(f"Metaphor contexts: {len(result['metaphor_contexts'])}")
        
        print("\nAI Terms:")
        for term in result['ai_terms']:
            print(f"  - {term['text']} ({term['pos']})")
        
        print("\nMetaphor Contexts:")
        for context in result['metaphor_contexts']:
            print(f"  - Pattern '{context['pattern']}' in: {context['sentence'][:100]}...")
        
        # Test chunked preprocessing
        print("\nTesting chunked preprocessing:")
        chunks = preprocess_for_chunked_analysis(sample_text, chunk_size=100)
        print(f"Created {len(chunks)} chunks")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
