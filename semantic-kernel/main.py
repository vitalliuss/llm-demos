import asyncio
import json
import sqlite3

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory, FunctionCallContent, FunctionResultContent
from semantic_kernel.functions import KernelArguments, kernel_function

DB_PATH = "db/movies.sqlite"
database_schema_string = """
Table: movies
Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
Table: directors
Columns: name, id, gender, uid, department
"""


class DatabasePlugin:
    @kernel_function(description="This function will return the information from the database.")
    def ask_database(self, sql_query: str) -> str:
        """Executes a SQL query against the SQLite database and returns results."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            print('SQL to be executed: ' + sql_query)
            cursor.execute(sql_query)
            results = cursor.fetchall()
            print(results)
            conn.close()
            return json.dumps(results)
        except Exception as e:
            return str(e)


kernel = Kernel()
kernel.add_plugin(DatabasePlugin(), plugin_name="database")

service_id = "agent"
kernel.add_service(OpenAIChatCompletion(service_id=service_id,
                                        ai_model_id="gpt-4o"))

settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
# Configure the function choice behavior to auto invoke kernel functions
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

# Define the agent name and instructions
AGENT_NAME = "MovieAgent"
AGENT_INSTRUCTIONS = ("If the question is related to movies, use the 'ask_database' tool. It can execute SQL against "
                      "the SQLite database and return results. Here is the database schema: ") + database_schema_string
# Create the agent
agent = ChatCompletionAgent(
    service_id=service_id,
    kernel=kernel,
    name=AGENT_NAME,
    instructions=AGENT_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)


async def main():
    chat_history = ChatHistory()

    user_input = "Show me 10 most expensive movies."
    chat_history.add_user_message(user_input)
    print(f"# User: '{user_input}'")

    agent_name: str | None = None
    print("# Assistant: ", end="")
    async for content in agent.invoke_stream(chat_history):
        if not agent_name:
            agent_name = content.name
            print(f"{agent_name}: '", end="")
        if (
                not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in content.items)
                and content.content.strip()
        ):
            print(f"{content.content}", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
