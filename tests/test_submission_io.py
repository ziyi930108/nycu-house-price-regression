"""Submission integrity checks independent of fitted models."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from submission_io import rounded_prices,write_submission

class SubmissionTests(unittest.TestCase):
    def test_numpy_ties_even_positive_floor(self):
        self.assertEqual(rounded_prices([0,-5,1.5,2.5,3.5]).tolist(),[1,1,2,2,4])
    def test_nonfinite_rejected(self):
        for p in [[float("nan")],[float("inf")]]:
            with self.assertRaises(ValueError):rounded_prices(p)
    def test_order_mismatch_fails_before_creation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"out.csv"
            with self.assertRaises(ValueError):write_submission(p,["1","2"],["2","1"],[4,5])
            self.assertFalse(p.exists())
    def test_bom_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"out.csv"
            write_submission(p,["1","2"],["1","2"],[1.2,2.9])
            self.assertTrue(p.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(p.read_bytes().count(b"\r\n"),3)
            with self.assertRaises(FileExistsError):write_submission(p,["1","2"],["1","2"],[4,5])
if __name__=="__main__":unittest.main()
