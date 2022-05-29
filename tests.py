import unittest

import numpy as np

import main
from model.dataset import load_mappings


class TestCase(unittest.TestCase):

    def testMain_HelloWorld(self):
        expected = "Hello World"
        self.assertEqual(main.main(), expected)

    def testLoadMappings_MediumDataset(self):
        expected = np.array(
            [
                "International", "Blues", "Jazz", "Classical",
                "Old-Time / Historic", "Country", "Pop", "Rock",
                "Easy Listening", "Soul-RnB", "Electronic",
                "Folk", "Spoken", "Hip-Hop", "Experimental",
                "Instrumental",
            ]
        )
        self.assertEqual(load_mappings().tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
