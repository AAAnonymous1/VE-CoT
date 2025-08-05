"""
gradio chat client with agent function calling
"""
import asyncio
from typing import Optional, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import re, json
from dataset_prompt import format_prompt
import gradio as gr
from PIL import Image
import time
from typing import Dict, Any, Generator
import logging
from pathlib import Path

# Prepare the python interpreter for the server virtual environment
python_dict = {"mcp_vision_server":"your_server_python_path",
               "mcp_search_server":"your_server_python_path"}

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
    match structured output，return SOAP text if output，otherwise return None
    Args:
        text:Textual content of LLM output
    """
    match = re.search(
        r"<think>\s*(.*?)\s*</think>.*?<S>\s*(.*?)\s*</S>.*?<O>\s*(.*?)\s*</O>.*?<A>\s*(.*?)\s*</A>.*?<P>\s*(.*?)\s*</P>",
        text,
        re.DOTALL,
    )
    
    return match.groups() if match is not None else None

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

def create_function_call_text(fc_tuple):
    """
    create function calling content html for gradio[unused]
    Args:
        fc_tuple: regex function calling from structured function calling
    """
    fc_str = ""
    for i, fc in enumerate(fc_tuple):
        if i == 0:
            fc_str += f"<div class=\"think\">{fc.replace("\n","<br>")}</div>"
            # 只要think内容
            return fc_str
        elif i == 1:
            fc_str += f"MCP服务器名称:{fc}<br>"
        elif i == 2:
            fc_str += f"工具名称:{fc}<br>"
        elif i == 3:
            fc_str += f"参数:{fc}<br>"
        else:
            break
    return fc_str


def create_output_text(output_tuple):
    """
    create output SOAP content html for gradio
    Args:
        output_tuple:regex output from structured output
    """
    output_think = ""
    output_soap = ""
    for i, output in enumerate(output_tuple):
        if i == 0:
            output_think += f"<div class=\"think\">{output.replace("\n","<br>")}</div>"
        elif i == 1:
            output_soap += f"S(Subject 情报背景):{output}<br>"
        elif i == 2:
            output_soap += f"O(Objective 客观数据):{output}<br>"
        elif i == 3:
            output_soap += f"A(Assessment 分析评估):{output}<br>"
        elif i == 4:
            output_soap += f"P(Plan 行动计划):{output}<br>"
        else:
            break
    return output_think, output_soap


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
    print(tool_call_list[0])
    return tool_call_list

def create_temp_folder(image_path):
    """
    create temp folder for intermediate image storage
    """
    file_path = Path(image_path)
    parent_dir = file_path.parent
    temp_dir = parent_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    print(f"Temp folder created：{temp_dir}") 

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
            server_script_path: Path to the server script(.py or .js)
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
            print(f"tool_args:{tool_call.tool_args}")
            result = await current_session.call_tool(tool_call.tool_name, tool_call.tool_args)
            result_list.append(result)
        return result_list

    async def cleanup(self):
        """
        close connections
        """
        await self.exit_stack.aclose()

async def run_vision_agent(user_input: Dict[str, Any], history: list, all_history):
    """
    run vision agent
    """
    image_path = user_input["files"][0]
    input_text = user_input["text"]
    create_temp_folder(image_path)
    mcp_client = MCPClient()
    user_messages = [{"role":"user", "content":input_text},{"role":"user","content":gr.Image(value=image_path)}]
    history.extend(user_messages)
    yield history, None
    openai_client = OpenAI(api_key="your_openai_api_key")
    try:
        print("Initializing connection")
        server_dict = {"mcp_vision_server":"your_server_.py_file_path",
                       "mcp_search_server":"your_server_.py_file_path"}
        await mcp_client.connect_to_server(server_dict)
        tool_dict = await mcp_client.get_tools()
        dataset_prompt = await format_prompt(tool_dict, "chat")

        print("Connection established")
        print("Processing decription")
        response = await mcp_client.call_tools([ToolCall(tool_server="mcp_vision_server", tool_name="vlm_describe", tool_args={"image_path":image_path, "text_prompt":"rough"})])
        response = json.loads(response[0].content[0].text)["vlm_response"]
        print("Description finished")
        messages = [{"role": "system", "content": [{"type":"text", "text":dataset_prompt}]},{"role":"assistant", "content":[{"type":"text", "text":response}]},{"role":"user", "content":[{"type":"text", "text":f"请根据VLM输出的描述按照要求调用工具，当前图像路径为{image_path}，当前任务请注意以下内容{input_text}"}]}]
        memory = messages
        while not check_structured_output(response):
            print("Requesting GPT")
            response = openai_client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=memory
            )
            print("Request finished")
            response = response.choices[0].message.content
            memory.append({"role":"assistant", "content":[{"type":"text", "text": response}]})
            if check_structured_function_call(response):
                print("Functin calling detected, process function calling")
                history.append({"role":"assistant", "content":create_function_call_text(check_structured_function_call(response))})
                yield history, None
                tool_call_list = create_call_query(response)
                tool_response = await mcp_client.call_tools(tool_call_list)
                try:
                    image = Image.open(json.loads(tool_response[0].content[0].text)["result_image_path"])
                    history.append({"role":"assistant", "content":gr.Image(image)})
                    yield history, None
                except Exception as e:
                    print("no image output")
                # history.append({"role":"assistant","content":json.loads(tool_response[0].content[0].text)["vlm_response"]})
                # yield history
                memory.append({"role":"assistant", "content":[{"type":"text", "text":tool_response[0].content[0].text}]})
                print("Function calling finished")
            elif check_structured_output(response):
                print("Task finished")
                output_think,output_soap = create_output_text(check_structured_output(response))
                final_output = user_messages
                final_output.append({"role":"assistant","content":output_soap})
                history = [msg for msg in history if msg["role"]=="assistant"]
                history.append({"role":"assistant","content":output_think})
                yield final_output, history
                print(final_output)
                break
            else:
                memory.append({"role":"user","content":[{"type":"text","text":"请继续探索，使用工具或是输出结果"}]})
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":

    with gr.Blocks(fill_height=True, css="""
        .think {
        background: #e0e0e0; color: #333;
        padding: 8px; border-radius: 6px; margin-bottom: 6px;
        }
        img {
        float: left;
        margin-right: 10px;
        }
    """) as demo:
        user_input = gr.State()  # 这里用来缓存输入文本
        archive_state = gr.State([])
        with gr.Accordion("点击查看推理过程", open=False):
            archive_chatbot = gr.Chatbot(type="messages", height=300, label="推理历史")
        chatbot = gr.Chatbot(type="messages",height=None, scale=1)
        textbox = gr.MultimodalTextbox(
            placeholder="上传图像+输入文本",
            file_types=["image"],
            file_count="single",
            sources=["upload"]
        )
        clear_event = textbox.submit(
            lambda x: (x, gr.update(value="正在运行，请稍等", interactive=False)),  # 立即清空
            inputs=textbox,
            outputs=[user_input, textbox]
        )

        clear_event.then(
            fn=run_vision_agent,
            inputs=[user_input, chatbot, archive_state],
            outputs=[chatbot, archive_state],
            queue=True
        ).then(
            lambda archive:archive,
            inputs=archive_state,
            outputs=archive_chatbot
        ).then(
            lambda x: (x, gr.update(value="", interactive=True)),
            inputs=textbox,
            outputs=[user_input, textbox]
        )

        
    demo.launch(allowed_paths=[""])