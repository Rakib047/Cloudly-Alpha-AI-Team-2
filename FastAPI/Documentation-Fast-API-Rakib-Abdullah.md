# FastAPI 

## Basic Things Before Going to FastAPI

### 1. Python Types

### What Are Python Type Hints?
Type hints (or type annotations) in Python let you declare the type of a variable or function return. This helps editors provide better autocompletion and error-checking without changing how the code runs.

### Example:
```python
def greet(name: str) -> str:
    return f"Hello, {name}"
````

* **str** here indicates the expected type for the `name` parameter and the return value.

### Benefits of Type Hints:

* Better autocompletion in editors.
* Helps detect errors early (like passing an integer instead of a string).

---

## 2. Asynchronous Programming (`async` / `await`)

### What is Asynchronous Code?

Asynchronous code allows the program to perform other tasks while waiting for slow operations (like network requests, file reading) to finish. This makes programs more efficient and responsive.

### Key Concepts:

* **`async def`**: Declares an asynchronous function.
* **`await`**: Pauses the function until the task completes, allowing other tasks to run in the meantime.

### Example:

````python
async def fetch_data():
    data = await some_network_call()
    return data
````

### Benefits:

* Non-blocking I/O operations.
* Handles many requests simultaneously without multiple threads.

---

### 3. Environment Variables

### What Are Environment Variables?

Environment variables are system settings that store configuration values outside of your Python code. They are useful for storing sensitive information, like API keys, or for configuration settings.

### Creating and Using Environment Variables:

1. Create a variable in the terminal:

````bash
export MY_KEY="secret"
````

2. Read it in Python:

````python
import os

api_key = os.getenv("MY_KEY", "default_key")
print(api_key)
````

### Temporary Variables:

Set a variable just for a program run:

````bash
MY_KEY="temporary_value" python main.py
````

---

### 4. Virtual Environments

### What is a Virtual Environment?

A virtual environment is a directory that contains a separate Python installation and its packages. It isolates your project from global Python settings, allowing you to have different dependencies for each project.

### Creating and Using a Virtual Environment:

1. Create a project directory:

````bash
mkdir awesome-project
cd awesome-project
````

2. Create a virtual environment:

````bash
python -m venv .venv
````

3. Activate the environment:

   * **Linux/macOS**:

   ````bash
   source .venv/bin/activate
   ````

   * **Windows PowerShell**:

   ````powershell
   .venv\Scripts\Activate.ps1
   ````

4. Install packages in the environment:

````bash
pip install fastapi
````

### Deactivating the Virtual Environment:

When done, deactivate the environment:

````bash
deactivate
````

---



