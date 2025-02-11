import json
import sqlite3
import requests
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
JIRA_API_URL= "https://your-company.atlassian.net/rest/api/2/issue/"
JIRA_AUTH = ("your_email", "your_api_token")
DB_PATH = "db/movies.sqlite"
database_schema_string = """
Table: movies
Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
Table: directors
Columns: name, id, gender, uid, department
"""

@tool
def ask_database(sql_query: str) -> str:
    """Executes a SQL query against the SQLite database and returns results."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return json.dumps(results)
    except Exception as e:
        return str(e)


@tool
def create_support_ticket(summary: str, description: str) -> str:
    """Creates a support ticket in Jira."""
    payload = {
        "fields": {
            "project": {"key": "SUP"},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"}
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(JIRA_API_URL, auth=JIRA_AUTH, json=payload, headers=headers)
        return response.text
    except Exception as e:
        return str(e)


tools = [ask_database, create_support_ticket]

llm_with_tools = llm.bind_tools(tools)

query = "Can you create a support ticket for me? I want to add movies released in 2024."
system_message = SystemMessage('If the question is related to movies, use ask_database tool. It can execute SQL '
                               'against the SQLite database and return results. Here is the database schema: ' +
                               database_schema_string +
                               'You can also create a support tickets using create_support_ticket tool if needed. '
                               'Construct the summary and description of the ticket based on the previous user '
                               'message and call the tool. Do not ask user for additional information. Resulting JIRA '
                               'issue url should be returned with the following format: https://darkkernel.atlassian.net/browse/SUP-123')
messages = [system_message,
            HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
print(ai_msg)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"ask_database": ask_database, "create_support_ticket": create_support_ticket}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print('\n tool_msg', tool_msg)
    messages.append(tool_msg)

result = llm_with_tools.invoke(messages)
print(result)
print('\n FINAL RESULT:', result.content)