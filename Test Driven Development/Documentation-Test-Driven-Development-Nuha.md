
# Test Driven Development in Python

A guide to different types of testing in Python, with examples, especially for Machine Learning workflows.

---

## Unit Testing

Unit tests focus on validating individual units of code (typically functions or methods). They are the foundation of a robust test suite.

### Key Characteristics:
- Fast and isolated from external dependencies.
- Ideal for testing utility functions and logic.
- Encourages modular code design.

### Example using `unittest`
```python
import unittest

def square(x):
    return x * x

class TestSquareFunction(unittest.TestCase):
    def test_square_positive(self):
        self.assertEqual(square(4), 16)

    def test_square_zero(self):
        self.assertEqual(square(0), 0)

    def test_square_negative(self):
        self.assertEqual(square(-3), 9)

if __name__ == '__main__':
    unittest.main()
```

### Example using `pytest`
```python
def square(x):
    return x * x

def test_square():
    assert square(5) == 25
    assert square(-2) == 4
```

### Running Tests
- `unittest`: `python -m unittest path/to/test_file.py`
- `pytest`: `pytest path/to/test_file.py`

### Common Assertions
- `assertEqual(a, b)` — checks equality
- `assertTrue(condition)` — asserts condition is true
- `assertRaises(Exception)` — checks if an exception is raised

---

## Integration Testing

Tests interactions between modules or services.

### Key Characteristics:
- Combines components to ensure they work together.
- Often involves database access, APIs, or file systems.
- May require setup and teardown methods.

### Example:
```python
def preprocess(data):
    return [x.lower() for x in data]

def tokenize(data):
    return [x.split() for x in data]

def full_pipeline(data):
    return tokenize(preprocess(data))

def test_pipeline_integration():
    raw = ["Hello World", "TEST case"]
    expected = [["hello", "world"], ["test", "case"]]
    assert full_pipeline(raw) == expected
```

---

## System Testing

End-to-end testing of the complete system in a production-like environment.

### Key Characteristics:
- Validates real-world workflows.
- Simulates user behavior.
- Often automated with tools like Selenium or Playwright.

### Example Use Case:
- Test an ML web dashboard: login → upload data → run model → view results.

---

## Regression Testing

Ensures new changes don’t break existing functionality.

### Key Characteristics:
- Run frequently in CI/CD pipelines.
- Can include unit, integration, and system tests.
- Prevents reintroducing old bugs.

### Example Use Case:
- After changing model architecture, confirm evaluation metrics remain consistent on validation data.

---

## Acceptance Testing

Validates that features meet business or user requirements.

### Key Characteristics:
- Derived from user stories or specs.
- May be manual or automated.
- Final sign-off for features.

### Example:
- User story: "As a user, I should see an error if no file is uploaded."
- Acceptance test confirms that the error is triggered correctly.

---

## Using Mocks in Tests

Mocking isolates the code under test from external dependencies.

### Key Characteristics:
- Replace APIs, DBs, file I/O, etc.
- Used in unit and integration tests.
- Makes tests faster and more predictable.

### Example with `unittest.mock`
```python
from unittest.mock import patch

def get_price():
    # Suppose this makes a real API call
    pass

@patch('__main__.get_price')
def test_price_fetch(mock_get):
    mock_get.return_value = 123.45
    assert get_price() == 123.45
```

### `patch('__main__.get_price')` temporarily replaces the get_price function with a mock object during the test.
---

## Setting Seeds for ML Testing

Reproducibility is essential when testing ML models.

### Why Set Seeds?
- Models involve randomness: initialization, dropout, shuffling.
- Seed ensures consistent test results.

### Example:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
```

---

## Summary

| Test Type         | Scope                | Purpose                                         |
|------------------|----------------------|--------------------------------------------------|
| Unit Test        | Single function/class| Check correctness in isolation                   |
| Integration Test | Multiple components  | Validate interactions between parts              |
| System Test      | Full system          | End-to-end validation                            |
| Regression Test  | Existing features    | Prevent bugs from resurfacing                    |
| Acceptance Test  | Business features    | Ensure requirements are met                      |

