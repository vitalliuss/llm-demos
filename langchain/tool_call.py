import json
import sqlite3
import requests
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv


load_dotenv()
token = os.getenv("GITHUB_TOKEN")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
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
def create_github_issue(title, body):
    """
    Creates an issue in the given GitHub repository and returns its title and URL.

    :param title: Issue title
    :param body: Issue body content
    :return: Tuple (issue_title, issue_url) or None if creation failed
    """
    url = f"https://api.github.com/repos/vitalliuss/llm-demos/issues"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    data = {
        "title": title,
        "body": body
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        issue_data = response.json()
        issue_title = issue_data["title"]
        issue_url = issue_data["html_url"]
        return issue_title, issue_url
    else:
        print(f"Failed to create issue: {response.status_code}")
        print(response.json())
        return None

tools = [ask_database, create_github_issue]

llm_with_tools = llm.bind_tools(tools)

# query = "Can you create a support ticket for me? I want to add movies released in 2024."
query = "I want to add new movies released in 2024 to the database. Can you help me with that?"
system_message = SystemMessage('If the question is related to movies, use ask_database tool. It can execute SQL '
                               'against the SQLite database and return results. Here is the database schema: ' +
                               database_schema_string +
                               ' You can also create a support tickets using create_github_issue tool if needed. '
                               'Construct the summary and description of the ticket based on the previous user '
                               'message and call the tool. Do not ask user for additional information.  '
                               )
messages = [system_message,
            HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
print(ai_msg)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"ask_database": ask_database, "create_github_issue": create_github_issue}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print('\n tool_msg', tool_msg)
    messages.append(tool_msg)

result = llm_with_tools.invoke(messages)
print(result)
print('\n FINAL RESULT:', result.content)