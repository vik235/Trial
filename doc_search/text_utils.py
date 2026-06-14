##--------------------------------------------------
##
## Takes the string input and tokenizes it into a list of words, removing stop words and punctuation.
## Example usage: tokens = tokenize("This is an example sentence, to demonstrate tokenization!")
## Output: ['example', 'sentence', 'demonstrate', 'tokenization']
##---------------------------------------------------

from typing import List
import logging 
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

_STOP_WORDS = set([
    'the', 'is', 'in', 'and', 'to', 'of', 'a', 'that', 'it', 'with', 
    'as', 'for', 'was', 'on', 'are', 'by', 'this', 'be', 'or', 'a',  'an',
    # Add more stop words as needed
])

_TOKENIZER_REGEX = re.compile(r'\b\w+\b')

def tokenize(text: str) ->  List[str]:
    """
    Tokenizes the input text into a list of words.
    """
    if not isinstance(text, str):
        logger.error("Input must be a string.")
        return []    
    if not text.strip():
        logger.warning("Input text is empty.")
        return []    
    # Use regex to find words, ignoring punctuation
    
    raw_tokens = _TOKENIZER_REGEX.findall(text.lower())
    tokens = [t for t in raw_tokens if t not in _STOP_WORDS]    
    logger.info(f"Tokenized text into {len(tokens)} tokens.")
    return tokens