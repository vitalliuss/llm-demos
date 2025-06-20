import time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="localhost-tools",
    host="0.0.0.0",
    port=8000
)

@mcp.tool(description="Get current date and time in %Y-%m-%d %H:%M:%S format")
async def get_current_datetime() -> str:
    datetime_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return  datetime_string

@mcp.tool(description="Convert Celsius to Fahrenheit")
async def convert_celsius_to_fahrenheit(celsius: float) -> str:
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

@mcp.tool(description="Convert Fahrenheit to Celsius")
async def convert_fahrenheit_to_celsius(fahrenheit: float) -> str:
    celsius = (fahrenheit - 32) * 5/9
    return f"{fahrenheit}°F is {celsius}°C"

@mcp.prompt(description="Summarize text to N sentences")
def summarize_text_to_n_sentences(text: str, sentences: int) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text into {sentences} sentences:\n\n{text}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.settings.log_level="DEBUG"
    mcp.run(transport="streamable-http") # or mcp.run(transport="sse") for Github Copilot Chat compatibility