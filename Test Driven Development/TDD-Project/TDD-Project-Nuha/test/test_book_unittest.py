import unittest
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go up to TDD-Project-Nuha/
sys.path.insert(0, str(project_root))

from src.Book_Class import Book



class TestBook(unittest.TestCase):

    def setUp(self):
        self.book = Book()

    def test_default_values(self):
        self.assertEqual(self.book.title, "Unknown")
        self.assertEqual(self.book.quantity, 0)
        self.assertEqual(self.book.author, "Unknown")
        self.assertEqual(self.book.price, 0.0)

    def test_repr_output(self):
        expected = "Book: Unknown, Quantity: 0, Author: Unknown, Price: 0.0"
        self.assertEqual(repr(self.book), expected)

    def test_assign_custom_values(self):
        self.book.title = "1984"
        self.book.author = "George Orwell"
        self.book.quantity = 10
        self.book.price = 12.99
        self.assertEqual(self.book.title, "1984")
        self.assertEqual(self.book.author, "George Orwell")
        self.assertEqual(self.book.quantity, 10)
        self.assertEqual(self.book.price, 12.99)

    def test_repr_with_custom_values(self):
        self.book.title = "1984"
        self.book.author = "George Orwell"
        self.book.quantity = 10
        self.book.price = 12.99
        expected = "Book: 1984, Quantity: 10, Author: George Orwell, Price: 12.99"
        self.assertEqual(repr(self.book), expected)

if __name__ == '__main__':
    unittest.main()

