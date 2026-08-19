# Using TypedDict for Structured Output

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    "name": "Arif",
    "age": 30
}

print(new_person)