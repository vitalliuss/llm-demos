import time
from openai import OpenAI

client = OpenAI()

USER_REQUEST = 'Give me the list of first 10 presidents of the United States of America.'

start_time = time.time()

response = client.chat.completions.create(
    model='gpt-4o',
    temperature=0,
    stream=True,
    messages=[
        {'role': 'user',
         'content': USER_REQUEST},
    ]
)

# create variables to collect the stream of chunks
collected_chunks = []
collected_messages = []
# iterate through the stream of events
for chunk in response:
    chunk_time = time.time() - start_time  # calculate the time delay of the chunk
    collected_chunks.append(chunk)  # save the event response
    chunk_message = chunk.choices[0].delta.content  # extract the message
    collected_messages.append(chunk_message)  # save the message
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text

# print the time delay and text received
print(f"Full response received {chunk_time:.2f} seconds after request")
# clean None in collected_messages
collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = ''.join(collected_messages)
print(f"Full conversation received: {full_reply_content}")
