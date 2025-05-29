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

## Basic Setup and Parameters in FastAPI

### Initial Setup

2. **Install FastAPI and Uvicorn**:

   ````bash
   pip install fastapi uvicorn
   ````

3. **Create `main.py`**:

   ````python
   from fastapi import FastAPI

   app = FastAPI()

   @app.get("/")
   async def read_root():
       return {"message": "Hello World"}
   ````

4. **Run the Application**:

   ````bash
   uvicorn main:app --reload
   ````

   Access the interactive API docs at `http://127.0.0.1:8000/docs`.

### Path Parameters

Path parameters allow you to capture values from the URL path.

**Example**:

````python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
````

* Accessing `http://127.0.0.1:8000/items/42` returns `{"item_id": 42}`.
* FastAPI automatically validates the type; accessing `http://127.0.0.1:8000/items/foo` will result in a validation error.

### Query Parameters

Query parameters are optional parameters appended to the URL after a `?`.

**Example**:

````python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
````

* Accessing `http://127.0.0.1:8000/items/?skip=5&limit=20` returns `{"skip": 5, "limit": 20}`.

**Optional Query Parameters**:

````python
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
````

* Accessing `http://127.0.0.1:8000/items/42?q=search` returns `{"item_id": 42, "q": "search"}`.

**Boolean Query Parameters**:

````python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, short: bool = False):
    item = {"item_id": item_id}
    if not short:
        item["description"] = "This is an amazing item with a long description"
    return item
````

* Accessing `http://127.0.0.1:8000/items/42?short=true` returns the item without the description.

---


Here’s a concise summary for your documentation on **Request Body in FastAPI** with key code examples:

---

## Request Body in FastAPI

* Use **request bodies** to receive data sent from clients (usually in POST, PUT, PATCH requests).
* Declare request bodies with **Pydantic models** to get automatic data parsing, validation, and editor support.

### Define a Pydantic model

````python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None  # Optional
    price: float
    tax: float | None = None        # Optional
````

### Use model as request body parameter

````python
@app.post("/items/")
async def create_item(item: Item):
    return item
````

* FastAPI reads JSON body, validates it, and converts data to correct types.
* Returns detailed errors if validation fails.

### Access model data inside function

````python
@app.post("/items/")
async def create_item(item: Item):
    item_data = item.dict()
    if item.tax:
        item_data["price_with_tax"] = item.price + item.tax
    return item_data
````

### Combine with path and query parameters

````python
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result["q"] = q
    return result
````

* FastAPI automatically detects source of parameters:

  * Path parameters from URL path
  * Request body from JSON body (Pydantic model)
  * Query parameters from URL query string

---




