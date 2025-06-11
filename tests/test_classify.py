import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from marcia_streamlit_app import classify_article


class TestClassifyArticle(unittest.TestCase):
    def test_opinion(self):
        self.assertEqual(classify_article("Opinion: Climate policy", ""), "opinion")

    def test_econpol(self):
        self.assertEqual(
            classify_article("Gobierno presenta nueva regulación", ""), "econpol"
        )

    def test_coyuntural(self):
        self.assertEqual(classify_article("Se inaugura la COP30", ""), "coyuntural")


if __name__ == "__main__":
    unittest.main()
