import os
import openai
import requests
import sqlite3
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from conversation import Conversation


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-3.5-turbo-0613"
GPT_4 = "gpt-4-0613"
DATABASE = "db/Chinook.db"
# DATABASE = "db/movies.sqlite"
USER_MESSAGE = "Hi, who are the top 10 artists by number of tracks"
# USER_MESSAGE = "Hi, what are the top 10 movies with the budget more than 10000000 by user rating"

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


if __name__ == '__main__':
    conversation = Conversation()
    conn = sqlite3.connect(DATABASE)


def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names

def get_table_rows_count(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchall()
    for col in columns:
        column_names.append(col[0])
    return column_names

def save_database_info_to_csv(conn, filename):
    get_database_info(conn)
    with open(filename, "w") as f:
        for table in get_database_info(conn):
            f.write(f"Table: {table['table_name']}\n")
            f.write(f"Columns: {', '.join(table['column_names'])}\n\n")



def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts



database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

print(f"Database schema string: '{database_schema_string}'")

functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about data. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]


def ask_database(conn, query):
    """Function to query SQLite database with provided SQL query."""
    try:
        results = conn.execute(query).fetchall()
        return results
    except Exception as e:
        raise Exception(f"SQL error: {e}")


def chat_completion_with_function_execution(messages, functions=None):
    """This function makes a ChatCompletion API call and if a function call is requested, executes the function"""
    try:
        response = chat_completion_request(messages, functions)
        full_message = response.json()["choices"][0]
        if full_message["finish_reason"] == "function_call":
            print(f"Function generation requested, calling function")
            return call_function(messages, full_message)
        else:
            print(f"Function not required, responding to user")
            return response.json()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return response


def call_function(messages, full_message):
    """Executes function calls using model generated function arguments."""

    # We'll add our one function here - this can be extended with any additional functions
    if full_message["message"]["function_call"]["name"] == "ask_database":
        query = eval(full_message["message"]["function_call"]["arguments"])
        print(f"Prepped query is {query}")
        try:
            results = ask_database(conn, query["query"])
        except Exception as e:
            print(e)

            # This following block tries to fix any issues in query generation with a subsequent call
            messages.append(
                {
                    "role": "system",
                    "content": f"""Query: {query['query']}
The previous query received the error {e}. 
Please return a fixed SQL query in plain text.
Your response should consist of ONLY the SQL query with the separator sql_start at the beginning and sql_end at the end""",
                }
            )
            response = chat_completion_request(messages, model=GPT_4)

            # Retrying with the fixed SQL query. If it fails a second time we exit.
            try:
                cleaned_query = response.json()["choices"][0]["message"][
                    "content"
                ].split("sql_start")[1]
                cleaned_query = cleaned_query.split("sql_end")[0]
                #print(cleaned_query)
                results = ask_database(conn, cleaned_query)
                #print(results)
                print("Got on second try")

            except Exception as e:
                print("Second failure, exiting")

                print(f"Function execution failed")
                print(f"Error message: {e}")

        messages.append(
            {"role": "function", "name": "ask_database", "content": str(results)}
        )

        try:
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            print(e)
            raise Exception("Function chat request failed")
    else:
        raise Exception("Function does not exist and cannot be called")


agent_system_message = """You are DatabaseGPT, a helpful assistant who gets answers to user questions from the Database
Provide as many details as possible to your users
Begin!"""

sql_conversation = Conversation()
sql_conversation.add_message("system", agent_system_message)
sql_conversation.add_message(
    "user", USER_MESSAGE
)

chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history, functions=functions
)
try:
    assistant_message = chat_response["choices"][0]["message"]["content"]
except Exception as e:
    print(e)
    print(chat_response)

sql_conversation.add_message("assistant", assistant_message)
sql_conversation.display_conversation(detailed=True)
