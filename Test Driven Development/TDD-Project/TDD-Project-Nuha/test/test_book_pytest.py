import pytest
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go up to TDD-Project-Nuha/
sys.path.insert(0, str(project_root))

from src.Book_Class import Book

@pytest.fixture
def default_book():
    return Book()

def test_default_values(default_book):
    assert default_book.title == "Unknown"
    assert default_book.quantity == 0
    assert default_book.author == "Unknown"
    assert default_book.price == 0.0

def test_repr_default(default_book):
    assert repr(default_book) == "Book: Unknown, Quantity: 0, Author: Unknown, Price: 0.0"

def test_assign_custom_values(default_book):
    default_book.title = "1984"
    default_book.author = "George Orwell"
    default_book.quantity = 10
    default_book.price = 12.99

    assert default_book.title == "1984"
    assert default_book.author == "George Orwell"
    assert default_book.quantity == 10
    assert default_book.price == 12.99

def test_repr_custom_values(default_book):
    default_book.title = "1984"
    default_book.author = "George Orwell"
    default_book.quantity = 10
    default_book.price = 12.99
    assert repr(default_book) == "Book: 1984, Quantity: 10, Author: George Orwell, Price: 12.99"
