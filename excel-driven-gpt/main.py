import datetime
import os
import threading
import openai
import pandas
from dotenv import load_dotenv


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
DATA_FILE = 'data.xlsx'
DATA_OUTPUT_XLSX = 'data_output.xlsx'

def openai_call(system_message, input):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "" + system_message + ""},
            {"role": "user", "content": "" + input + ""}
        ]
    )
    print(response)
    result = response.choices[0].message.content
    return result


def api_call_with_timeout(system_message, input, timeout_seconds, default_response):
    response = [default_response]

    def make_api_call():
        try:
            response[0] = openai_call(system_message, input)
        except Exception as e:
            print(e)
            pass

    thread = threading.Thread(target=make_api_call)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        return default_response
    else:
        return response[0]


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    dataframe = pandas.read_excel(DATA_FILE)
    call_count = 0
    # iterate through the rows of Excel file
    for index, row in dataframe.iterrows():
        system_message = row['SYSTEM MESSAGE']
        user_message = row['USER']
        try:
            call_count += 1
            print('Call #' + str(call_count))
            ai_response = api_call_with_timeout(system_message, user_message, 30, -1) # seconds to wait for openai to respond
        except Exception as e:
            ai_response = e
        # write content to dataframe column and row
        dataframe.at[index, 'ASSISTANT'] = ai_response
    dataframe.to_excel(DATA_OUTPUT_XLSX, index=False)
    time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print('Time elapsed: ' + str(time_elapsed) + ' seconds')


