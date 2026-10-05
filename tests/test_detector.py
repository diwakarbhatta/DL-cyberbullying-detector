"""
Unit tests for preprocessing and toxicity detector
"""

import unittest
from src.preprocessor import clean_text, normalize_leetspeak

class TestPreprocessor(unittest.TestCase):
    def test_clean_text(self):
        raw = "Check this out: https://example.com/bad and @john is cool!!!"
        cleaned = clean_text(raw)
        self.assertIn("[URL]", cleaned)
        self.assertIn("[USER]", cleaned)
        self.assertIn("cool!!", cleaned)

    def test_normalize_leetspeak(self):
        obfuscated = "f*ck this sh!t and you b!tch"
        normalized = normalize_leetspeak(obfuscated)
        self.assertIn("fuck", normalized)
        self.assertIn("shit", normalized)
        self.assertIn("bitch", normalized)

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(normalize_leetspeak(""), "")

if __name__ == "__main__":
    unittest.main()
