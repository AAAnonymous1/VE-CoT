"""
mcp search tools server
"""
from tavily import TavilyClient
from mcp.server.fastmcp import FastMCP
import mcp.types as types
import asyncio

search_server = FastMCP("search tools")

@search_server.tool()
async def tavily_search(query: str):
    """
    use tavily search to conduct web search
    Args:
        query:search keyword or sentence
    """
    try:
        tavily_client = TavilyClient(api_key="your_tavily_api_key")
        response = tavily_client.search(query=query, max_results=3)
        res_list = []
        for res in response['results']:
            res_list.append(str(res))
    except Exception as error:
        return types.CallToolResult(
        isError=True,
        content=[
            types.TextContent(
                type="text",
                text=f"Error: {str(error)}"
            )
        ]
    )
    print(res_list)
    return {"result_list":res_list}

if __name__=="__main__":
    search_server.run(transport="stdio")