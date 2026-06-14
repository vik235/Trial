import unittest 

from doc_search.search_engine import SearchEngine
from doc_search.text_utils import tokenize


class TestTokenize(unittest.TestCase):
    def test_tokenize_basic(self):
        text = "This is an example sentence, to demonstrate tokenization!"
        expected_tokens = ['example', 'sentence', 'demonstrate', 'tokenization']
        tokens = tokenize(text)
        self.assertEqual(tokens, expected_tokens)    

    def test_tokenize_empty_string(self):
        text = ""
        expected_tokens = []
        tokens = tokenize(text)
        self.assertEqual(tokens, expected_tokens)

    def test_tokenize_only_stop_words(self):
        text = "the and is in"
        expected_tokens = []
        tokens = tokenize(text)
        self.assertEqual(tokens, expected_tokens)


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine()
        
    
    def test_add_document(self):
        doc = "This is a test document."
        doc_id = self.engine.add_document(doc)
        self.assertEqual(doc_id, 0)
        self.assertEqual(self.engine.documents[0], doc)
    
    def test_add_documents(self):
        docs = ["First document.", "Second document.", "Third document."]
        doc_ids = self.engine.add_documents(docs)
        self.assertEqual(doc_ids, [0, 1, 2])
        self.assertEqual(self.engine.documents, docs)

    def test_engine_size(self):
        self.assertEqual(self.engine.size, 0)
        self.engine.add_document("Doc 1")
        self.engine.add_document("Doc 2")
        self.assertEqual(self.engine.size, 2)
    
if __name__ == '__main__':
    unittest.main()
    