import os
import openai
import time
from dotenv import load_dotenv

USER_REQUEST = 'Give me the list of first 30 presidents of the United States of America.'

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


if __name__ == '__main__':
    # record the time before the request is sent
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user',
             'content': USER_REQUEST},
        ],
        temperature=0,
        stream=True
    )

    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        content = (chunk_message.get('content'))
        # print content if it is not None (i.e., if it is a message from the AI) and don't print the newline
        if content is not None:
            print(content, end='')

    print()
    print(f"Full response received {chunk_time:.2f} seconds after request")
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])

