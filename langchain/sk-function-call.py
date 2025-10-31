import asyncio
import json
import os
import sqlite3
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel import Kernel

DB_PATH = "db/movies.sqlite"
database_schema_string = """
Table: movies
Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
Table: directors
Columns: name, id, gender, uid, department
"""

class DatabasePlugin:
    @kernel_function(name="AskDatabase", description="Ask the database a question.")
    async def ask_database(self, query: str, arguments: KernelArguments) -> str:
        # The 'arguments' parameter is reserved and automatically populated with KernelArguments.
        print("Executing SQL Query: ", query)
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            return json.dumps(results)
        except Exception as e:
            return str(e)

async def main():
    kernel = Kernel()

    chat_completion_service = OpenAIChatCompletion(
        ai_model_id="gpt-4.1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    kernel.add_service(chat_completion_service)
    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    kernel.add_plugin(DatabasePlugin(), plugin_name="DatabasePlugin")

    chat_history = ChatHistory()
    chat_history.add_developer_message("You are an assistant that can get data from a movies database. Database schema: " + database_schema_string)
    chat_history.add_user_message("Get data from the database about top 10 movies released in 2007.")

    response = await chat_completion_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )

    print("Chat Completion Response: ", response)

if __name__ == "__main__":
    asyncio.run(main())