# Overview of the work

This document covers various types of software testing using Python's `unittest` framework. It includes **Unit Testing**, which ensures individual functions work correctly; **Integration Testing**, validating module interactions; **System Testing**, testing the full system integration; **Regression Testing**, ensuring recent changes haven’t broken existing functionality; **Acceptance Testing**, confirming software readiness for end-users; and the use of Mocking to simulate external dependencies. Additionally, it explains **Reusing Seed** for ML Testing to ensure reproducibility in machine learning experiments. Each section provides examples and instructions for running tests.

---
# Unit Testing

### Scope: 
Function

### Purpose: 
It ensures individual functions or methods work correctly.

```python
# calculator.py
class Calculator:
    def add(self, a, b):
        """Return the sum of a and b."""
        return a + b

# test_calculator.py
import unittest

class TestCalculator(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.calculator = Calculator()

    def test_add(self):
        """Test addition method."""
        self.assertEqual(self.calculator.add(1, 1), 2)
        self.assertEqual(self.calculator.add(-1, 1), 0)
        self.assertEqual(self.calculator.add(-1, -1), -2)
        
        # Test if adding strings raises a TypeError
        with self.assertRaises(TypeError):
            self.calculator.add("1", "1")  # Adding strings should raise an error

    def tearDown(self):
        """Clean up after each test."""
        del self.calculator

if __name__ == "__main__":
    unittest.main()

```

### How to Run:
To run the unit tests, navigate to the folder where your `test_calculator.py` file is located and use the following command in your terminal:

```bash
python3 -m unittest test_calculator.py
```

---

# Integration Testing
### Scope: 
Modules

### Purpose: 
It ensures different modules work together correctly. This testing identifies issues that might not be visible in isolated unit tests.

**Example**:
Imagine we have a login module and a database module. Integration testing would check if the login module is able to communicate with the database module and successfully retrieve and validate user credentials.

```python

import unittest

class DatabaseService:
    """Simulates a simple database for storing user data."""
    def __init__(self):
        self.users = []

    def save_user(self, username, email):
        """Save the user data in the database."""
        self.users.append({"username": username, "email": email})

    def get_user(self, username):
        """Retrieve a user from the database."""
        for user in self.users:
            if user["username"] == username:
                return user
        return None

class UserService:
    """Handles user registration."""
    def __init__(self, db_service):
        self.db_service = db_service

    def register_user(self, username, email):
        """Register a user by saving their data."""
        self.db_service.save_user(username, email)
        return "User registered successfully!"

class TestUserRegistration(unittest.TestCase):

    def setUp(self):
        """Set up the services before each test."""
        self.db_service = DatabaseService()
        self.user_service = UserService(self.db_service)

    def test_user_registration(self):
        """Test that user registration works and data is saved correctly."""
        # Register a new user
        self.user_service.register_user("john_doe", "john@example.com")
        
        # Check if the user data is stored in the database
        user = self.db_service.get_user("john_doe")
        
        self.assertIsNotNone(user)  # User should be found
        self.assertEqual(user["username"], "john_doe")
        self.assertEqual(user["email"], "john@example.com")

if __name__ == "__main__":
    unittest.main()
```
----

# System Testing

### Scope : 
Full System 

### Purpose : 
System testing involves testing the complete system as a whole to ensure it works as expected in all environments. It checks if the system meets its requirements and if all components integrate well together, including hardware, software, and interfaces.

### Example : 
After we test the integration of individual modules (like login, registration, and payment), system testing would check the entire application flow—from opening the app, logging in, making a purchase, and logging out. It ensures everything works together as a complete system.

```python
import unittest

class ECommerceApp:
    def __init__(self):
        self.logged_in = False
        self.items = {"item1": 100, "item2": 200}
        self.cart = {}

    def login(self, username, password):
        """Simulate login."""
        if username == "Rakib" and password == "helloWorld":
            self.logged_in = True
            return "Login successful"
        return "Login failed"

    def add_to_cart(self, item, quantity):
        """Add item to cart if logged in."""
        if self.logged_in and item in self.items:
            self.cart[item] = quantity
            return f"{item} added to cart"
        return "You must be logged in to add items to the cart"

    def checkout(self):
        """Simulate checkout process."""
        if not self.logged_in:
            return "You must be logged in to checkout"
        total = sum(self.items[item] * qty for item, qty in self.cart.items())
        return f"Total: ${total}"

    def logout(self):
        """Simulate logout."""
        self.logged_in = False
        return "Logout successful"


class TestECommerceApp(unittest.TestCase):

    def setUp(self):
        """Set up the E-commerce app for testing."""
        self.app = ECommerceApp()

    def test_system_flow(self):
        """Test the entire application flow (login, purchase, logout)."""
        # Test login
        self.assertEqual(self.app.login("Rakib", "helloWorld"), "Login successful")

        # Test adding items to cart
        self.assertEqual(self.app.add_to_cart("item1", 2), "item1 added to cart")
        
        # Test checkout
        self.assertEqual(self.app.checkout(), "Total: $200")  # 2 * 100 = 200
        
        # Test logout
        self.assertEqual(self.app.logout(), "Logout successful")
        
        # Test logged-out state
        self.assertEqual(self.app.add_to_cart("item2", 1), "You must be logged in to add items to the cart")


if __name__ == "__main__":
    unittest.main()
```

----

# Regression Testing

### Scope : 
Full System

### Purpose : 
Regression testing involves re-running previous tests to ensure that recent changes, enhancements, or bug fixes have not introduced new bugs or broken existing functionality.

```python
import unittest

class Calculator:
    def add(self, a, b):
        """Add two numbers (supporting integers and floats)."""
        return a + b

class TestCalculator(unittest.TestCase):

    def setUp(self):
        """Set up the Calculator instance before each test."""
        self.calculator = Calculator()

    def test_add_integers(self):
        """Test adding integers (regression test)."""
        self.assertEqual(self.calculator.add(2, 3), 5)  # 2 + 3 = 5

    def test_add_floats(self):
        """Test adding floats (new feature)."""
        self.assertEqual(self.calculator.add(2.5, 3.5), 6.0)  # 2.5 + 3.5 = 6.0

if __name__ == "__main__":
    unittest.main()
```

---

# Acceptance Testing

### Scope : 
User-level

### Purpose : 
Acceptance testing verifies whether the software meets the business requirements and if it is ready for use by the end-user. This type of testing is often performed by the client or the end-users to confirm the software’s functionality and usability.

# `@mock`

Mocking (using `@mock` in python) is used to simulate parts of the system that aren’t yet implemented or are too complex (like databases, external APIs) during testing. It allows us to isolate and test specific components without worrying about their dependencies.

```python
import unittest
from unittest.mock import patch

class UserService:
    def get_user_info(self, user_id):
        # Simulates calling an external API
        # return requests.get(f"https://api.example.com/users/{user_id}").json()
        return {"user_id": user_id, "name": "John Doe", "age": 30}

class TestUserService(unittest.TestCase):
    #@patch.object(UserService, 'get_user_info') is a mocking decorator that tells 
    # the test to replace the real get_user_info method in UserService with a mock version.
    @patch.object(UserService, 'get_user_info')
    def test_get_user_info(self, mock_get_user_info):
        """Test that the get_user_info function returns correct data."""
        # Define the mock return value
        mock_get_user_info.return_value = {"user_id": 1, "name": "Mock User", "age": 25}
        
        user_service = UserService()
        result = user_service.get_user_info(1)

        # Assert that the mock data is returned, not real API data
        self.assertEqual(result["name"], "Mock User")
        self.assertEqual(result["age"], 25)
        mock_get_user_info.assert_called_once_with(1)  # Ensure the function was called with the correct argument

if __name__ == "__main__":
    unittest.main()
```
---

# Reusing Seed for ML Testing

Reusing a fixed random seed ensures that the results of a machine learning model are **reproducible**. When training models or performing tests that involve randomness (random splits of data, initialization of weights etc), using a fixed seed allows us to get the same results every time you run the test.

This is crucial in machine learning, as it allows us to verify our results, debug issues, and compare different models or configurations accurately. Without a fixed seed, results may vary between runs, making it hard to draw reliable conclusions.

```python
import random
import numpy as np

# Set a fixed seed
def set_seed(seed=42):
    random.seed(seed)

# Function that generates random numbers
def generate_random_numbers():
    return random.randint(1, 100)

# Set the seed to ensure reproducibility
set_seed(43)

# Generate random numbers
random_num= generate_random_numbers()
print(f"Random int: {random_num}")
```

### Output

```bash
Random int: 5
```