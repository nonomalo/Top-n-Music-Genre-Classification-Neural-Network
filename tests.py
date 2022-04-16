import unittest
import main


class TestCase(unittest.TestCase):

    def test1(self):
        expected = "Hello World"
        self.assertEqual(main.main(), expected)


if __name__ == '__main__':
    unittest.main()
