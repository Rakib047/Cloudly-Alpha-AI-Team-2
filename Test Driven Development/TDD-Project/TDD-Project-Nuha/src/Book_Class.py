class Book:
    def __init__(self, title="Unknown", quantity=0, author="Unknown", price=0.0):
        self.title = title
        self.quantity = quantity
        self.author = author
        self.price = price

    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}, Price: {self.price}"