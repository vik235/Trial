from dataclasses import dataclass, field, asdict, astuple
import math
from typing import List, Dict, Any
import json


from doc_search.text_utils import tokenize

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Vector = Dict[str, float]

@dataclass
class SearchResult: 
    doc_id: int
    score: float
    text: str



@dataclass
class SearchEngine:
    _docs: List[str] = field(default_factory=list) # List of documents in the search engine example: ["Document 1 text", "Document 2 text", ...]    
    _docs_tokens: List[List[str]] = field(default_factory=list) # List of tokenized documents, where each document is represented as a list of tokens. Example: [["token1", "token2"], ["token3", "token4"], ...]
    _doc_vectors: List[Vector] = field(default_factory=list) # List of vector representations for each document, where each vector is a dictionary mapping terms to their TF-IDF weights. Example: [{"term1": 0.5, "term2": 0.3}, {"term3": 0.7, "term4": 0.2}, ...]
    _doc_ids: List[int] = field(default_factory=list) # List of unique identifiers
    _dirty: bool = False # Flag to indicate if the search engine state has changed and needs to be saved or updated. Example: True if new documents have been added or existing documents have been modified, False otherwise.  
    _idf: Dict[str, float] = field(default_factory=dict) # Dictionary mapping terms to their inverse document frequency (IDF) values. Example: {"term1": 1.5, "term2": 2.0, ...}



    def add_document(self, doc: str) -> int:
        """
        Adds a document to the search engine and returns its unique identifier.
        """
        if not isinstance(doc, str):
            logger.error("Document must be a string.")
            raise ValueError("Document must be a string.")
        if not doc.strip():
            logger.warning("Document is empty.")
            raise ValueError("Document cannot be empty.")
        
        doc_id = len(self._docs)
        self._docs.append(doc)
        self._docs_tokens.append(tokenize(doc))
        self._doc_ids.append(doc_id)
        self._dirty = True
        logger.info(f"Added document with ID {doc_id}.")
        return doc_id


    def add_documents(self, docs: List[str]) -> List[int]:
        """
        Adds multiple documents to the search engine and returns a list of their unique identifiers.
        """
        if not isinstance(docs, List):
            logger.error("Input must be a list of strings.")
            raise ValueError("Input must be a list of strings.")
        
        doc_ids = []
        for doc in docs:
            try:
                doc_id = self.add_document(doc)
                doc_ids.append(doc_id)
            except ValueError as e:
                logger.warning(f"Skipping document due to error: {e}")
                continue
        logger.info(f"Added {len(doc_ids)} documents.")
        return doc_ids
    
    @property
    def documents(self) -> List[str]:
        """
        Returns the list of documents in the search engine.
        """
        return self._docs
    
    @property
    def size(self) -> int:
        """
        Returns the number of documents in the search engine.
        """
        return len(self._docs)
    
    def _compute_idf(self) -> None:

        doc_freq = {}
        for tokens in self._docs_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1    

        self._idf = {term: math.log(len(self._docs) / df) + 1 for term, df in doc_freq.items()}

    def _vectorize(self, tokens: List[str]) -> Vector:
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        vector = {term: (tf.get(term, 0) / len(tokens)) * self._idf.get(term, 0) for term in self._idf}

        return {term: weight for term, weight in vector.items() if weight > 0}

    def _rebuild(self) -> None:
        self._compute_idf()
        self._doc_vectors = [self._vectorize(tokens) for tokens in self._docs_tokens]
        self._dirty = False
        logger.info("Rebuilt search engine vectors.")

    # Similarity and search methods would go here, but are not implemented in this snippet.
    #

    @staticmethod
    def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[term] * vec2[term] for term in intersection)
        sum1 = sum(weight ** 2 for weight in vec1.values())
        sum2 = sum(weight ** 2 for weight in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if denominator == 0:
            return 0.0
        else:
            return numerator / denominator
        

    def search(self, query, top_k: int = 5, *, min_score: float = 0.0) -> List[SearchResult]:
        if self._dirty:
            self._rebuild()
        
        query_tokens = tokenize(query)
        query_vector = self._vectorize(query_tokens)

        results = []
        for doc_id, doc_vector in zip(self._doc_ids, self._doc_vectors):
            score = self.cosine_similarity(query_vector, doc_vector)
            if score >= min_score:
                results.append(SearchResult(doc_id=doc_id, score=score, text=self._docs[doc_id]))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
