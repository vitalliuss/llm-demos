from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o",
  max_tokens=100,
  seed=42,
  temperature=0.8,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke"}
  ]
)

print(completion.choices[0].message)