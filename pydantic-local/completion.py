import openai


client = openai.Client(
    base_url="http://localhost:1234/v1/"
)

response = client.chat.completions.create(
    model="phi-4",
    temperature=0.5,
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the fastest car in the world?"},
    ]
)

print(response.choices[0].message.content)