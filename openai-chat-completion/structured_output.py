from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


class Person(BaseModel):
    name: str
    surname: str
    date_of_birth: str
    favourite_movies: list[str]


class Clients(BaseModel):
    persons: list[Person]


completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "Generate me a list of 5 persons with name, surname, date of birth and favourite movies."}
    ],
    response_format=Clients,
)

person = completion.choices[0].message.parsed
print(person)
