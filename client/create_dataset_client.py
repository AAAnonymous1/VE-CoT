"""
create dataset client with agent function calling
"""
import asyncio
from typing import Optional, List
from contextlib import AsyncExitStack

import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import re, json
from dataset_prompt import format_prompt, generate_tool_xml
import gradio as gr
from PIL import Image
import time
from typing import Dict, Any, Generator
import logging
import tiktoken
import random
from pathlib import Path

python_dict = {"mcp_vision_server":"your_server_python_path",
               "mcp_search_server":"your_server_python_path"}


def format_dataset(image_path, finetune_memory, dataset_list):
    """
    format dataset
    Args:
        image_path:processed image path
        finetune_memory:LLM output formatted memory
        dataset_list:dataset list
    """
    messages = []
    for i,conversation in enumerate(finetune_memory):
        if conversation['role'] == 'user' and i == 1:
            messages.append({"role":"user","content":[{"type":"text","text":conversation["content"]},{"type":"image","image":image_path}]})
        else:
            messages.append({"role":conversation['role'],"content":[{"type":"text","text":conversation["content"]}]})
    dataset_list.append({"id":Path(image_path).stem, "messages":messages})

    return dataset_list

def generate_user_prompt() -> str:
    """
    randomly generate some user input prompt
    """
    return f"当前图像为卫星遥感图像，拍摄于{random.randint(2002,2025)}年{random.randint(1,12)}月{random.randint(1,28)}号"


class ToolCall:
    """
    Tool call class for storing all parameters of a function call
    Args:
        tool_server:Name of the server to which the tool belongs
        tool_name:Tool name
        tool_args:Tool Call Parameters
    """
    def __init__(self, tool_server, tool_name, tool_args):
        self.tool_server = tool_server
        self.tool_name = tool_name
        self.tool_args = tool_args

    def __str__(self):
        return f"tool server={self.tool_server},tool name={self.tool_name},tool args={self.tool_args}"

def get_python_path_for_server(server_name):
    """
    Get the server virtual environment interpreter of the server name.
    Args:
        server_name:server name
    """
    if server_name in python_dict.keys():
        return python_dict[server_name]
    else:
        raise ValueError("No server found")

def check_structured_output(text: str):
    """
    match structured output，return text if output <end>，otherwise return None
    Args:
        text:Textual content of LLM output
    """
    match = re.search(
        r"<think>(.*?)</think>(.*?)<end>",
        text,
        re.DOTALL,
    )
    
    return True if match  else False

def check_structured_function_call(text: str):
    """
    match structured function call, return tool call args if match, otherwise return None
    Args:
        text:Textual content of LLM output
    """
    match = re.search(
        r"<think>\s*(?P<think>.*?)\s*</think>\s*"
        r"<use_mcp_tool>\s*"
        r"<server_name>(?P<server>.*?)</server_name>\s*"
        r"<tool_name>(?P<tool>.*?)</tool_name>\s*"
        r"<arguments>\s*(?P<args>\{.*?\})\s*</arguments>\s*"
        r"</use_mcp_tool>",
        text,
        re.DOTALL
    )
    return match.groups() if match is not None else None

def create_call_query(call_text):
    """
    match function calling args
    Args:
        call_text:LLM structured function calling
    """
    pattern = re.compile(
        r"<use_mcp_tool>\s*"
        r"<server_name>(?P<server>.*?)</server_name>\s*"
        r"<tool_name>(?P<tool>.*?)</tool_name>\s*"
        r"<arguments>\s*(?P<args>{.*?})\s*</arguments>\s*"
        r"</use_mcp_tool>",
        re.DOTALL
    )

    matches = pattern.finditer(call_text)
    tool_call_list = []
    for match in matches:
        tool_call = ToolCall(match.group("server"), match.group("tool"), json.loads(match.group("args")))
        tool_call_list.append(tool_call)
    return tool_call_list

    

class MCPClient:
    """
    MCP client definition
    """
    def __init__(self):
        self.session_dict: dict[str, Optional[ClientSession]] = {}
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_dict: dict):
        """connect to MCP server
        Args：
            server_path: Path to the server script(.py or .js)
        """
        for server in server_dict.items():
            is_python = server[1].endswith('.py')
            is_js = server[1].endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")
            command = get_python_path_for_server(server[0]) if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server[1]],
                env=None
            )
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await session.initialize()
            response = await session.list_tools()
            tools = response.tools
            self.session_dict[server[0]] = session
            print(f"\nConnected to server {server[0]} with tools:", [tool.name for tool in tools])

        
    async def get_tools(self):
        """
        request tools from servers and format a tool dict
        """
        tools_dict: dict[str, list] = {}
        for session in self.session_dict.items():
            response = await session[1].list_tools()
            tools = response.tools
            tools_dict[session[0]] = tools
        return tools_dict
    
    async def call_tools(self, tool_call_list: str):
        """
        call tools based on tool call list
        Args:
            tool_call_list:ToolCall class list
        """
        result_list = []
        for tool_call in tool_call_list:
            current_session = self.session_dict[tool_call.tool_server]
            result = await current_session.call_tool(tool_call.tool_name, tool_call.tool_args)
            result_list.append(result)
        return result_list

    async def cleanup(self):
        """
        close connections
        """
        await self.exit_stack.aclose()

async def run_vision_agent(image_path, dataset_list):
    """
    run vision agent, changed llm to qwen-plus due to tight budget
    you can easily change llm back to chatgpt or other providers
    """
    input_token = 0
    output_token = 0
    mcp_client = MCPClient()
    qwen_client = OpenAI(api_key="your_qwen_api_key", base_url="qwen_api_url")
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        logging.info("Initializing connection")
        server_dict = {"mcp_vision_server":"your_server_.py_file_path",
                       "mcp_search_server":"your_server_.py_file_path"}
        await mcp_client.connect_to_server(server_dict)
        tool_dict = await mcp_client.get_tools()
        dataset_prompt = await format_prompt(tool_dict,"dataset")
        tools_xml = generate_tool_xml(tool_dict)
        logging.info("Connection established")
        logging.info("Processing decription")
        response = await mcp_client.call_tools([ToolCall(tool_server="mcp_vision_server", tool_name="vlm_describe", tool_args={"image_path":image_path, "text_prompt":"rough"})])
        response = json.loads(response[0].content[0].text)["vlm_response"]
        logging.info("Description finished")
        messages = [{"role": "system", "content":dataset_prompt},{"role":"assistant", "content":response},{"role":"user", "content":f"请根据VLM输出的描述按照要求调用工具，当前图像路径为{image_path}"}]
        finetune_memory = [{"role": "system", "content": f"你是一个遥感图像分析师，你可以使用以下工具来辅助你的判断\n{tools_xml}\n\n请严格按照工具参数进行调用"},{"role":"user","content":generate_user_prompt()},{"role":"assistant", "content": response},{"role":"user", "content":f"请根据VLM输出的描述按照要求调用工具，当前图像路径为{image_path}"}]
        memory = messages
        while not check_structured_output(response):
            logging.info("Requesting qwen")
            response = qwen_client.chat.completions.create(
                model="qwen-plus",
                messages=memory,
                extra_body={"enable_thinking": False},
            )
            logging.info("Request finished")
            input_token += response.usage.prompt_tokens
            output_token += response.usage.completion_tokens
            response = response.choices[0].message.content
            memory.append({"role":"assistant", "content": response})
            finetune_memory.append({"role":"assistant", "content":response})
            if check_structured_function_call(response):
                logging.info("Functin calling detected, process function calling")
                tool_call_list = create_call_query(response)
                tool_response = await mcp_client.call_tools(tool_call_list)
                memory.append({"role":"assistant", "content":tool_response[0].content[0].text})
                finetune_memory.append({"role":"tool", "content":tool_response[0].content[0].text})
                logging.info("Function calling finished")
            elif check_structured_output(response):
                logging.info("Task finished")
                logging.info(f"input token:{input_token},outputtoken:{output_token}")
                return format_dataset(image_path, finetune_memory, dataset_list)
                break
            else:
                memory.append({"role":"user","content":[{"type":"text","text":"请继续探索，使用工具或是输出结果"}]})
    finally:
        await mcp_client.cleanup()

async def create_dataset():
    logging.basicConfig(
            level=logging.INFO,  # 设置最低输出等级为 INFO
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
    logging.info("Creating dataset")
    dataset_path = "your_path_to_dataset_images"

    logging.info("Reading processed images")
    processed_images = []
    if os.path.getsize("dataset.json") != 0:
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset_list = json.load(f)
    else:
        dataset_list = []
    for data in dataset_list:
        processed_images.append(data["id"])
    
    logging.info("Processing image")
    for i, image in enumerate(os.listdir(dataset_path)):
        if Path(image).suffix == ".json" or Path(image).is_dir():
            continue
        logging.info(f"Processing {image},No.{i}")
        if Path(image).stem in processed_images:
            logging.info(f"{image}is processed, skipping")
            continue
        image_path = os.path.join(dataset_path, image)
        dataset_list = await run_vision_agent(image_path, dataset_list)
        print(dataset_list[-1])
        with open("dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_list, f, ensure_ascii=False, indent=4)
        if i == 100:
            break
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    asyncio.run(create_dataset())
