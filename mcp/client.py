from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


async def main():
    # Connect to a streamable HTTP server
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            print("Session initialized.")
            # List available tools
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                print(f"Tool: {tool}")
            # Call a tool
            tool_result = await session.call_tool("get_current_datetime")
            print(f"Tool result: {tool_result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())