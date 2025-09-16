"""
call vlm api
"""
from openai import OpenAI
from prompts.vision_prompt import prepare_prompt
import mcp.types as types
from ollam import AsyncClient

client = OpenAI(api_key="your_openai_api_key")

def create_file(image_path):
    """
    create file through openai file api
    Args:
        input image path
    Return:
        openai file id
    """
    with open(image_path, 'rb') as image_content:
        result = client.files.create(
            file=image_content,
            purpose="vision"
        )
    return result.id

async def vlm_prepare_dataset(image_path: str, task_type: str, **kwargs) -> str:
    """
    call vlm during the construction of a dataset
    Args:
        image_path:input image path
        task_type:task type，rough or tool
        kwargs:extra text from tools，like GroundingDINO boxes text
    """
    try:
        file_id = create_file(image_path)
        prompt = prepare_prompt("dataset", task_type)
        text = prompt
        if "dino" in kwargs:
            dino = kwargs["dino"]
            text += f"GroundingDINO boxes:\n{dino}"
        if "text_prompt" in kwargs:
            text_prompt = kwargs["text_prompt"]
            text += f"\n请按照以下提示词重点详细描述图像:{text_prompt}"

        response = client.responses.create(
            model="chatgpt-4o-latest",
            input=[{
                "role":"user",
                "content":[
                    {
                        "type":"input_text",
                        "text":text
                    },
                    {
                        "type":"input_image",
                        "file_id":file_id
                    }
                ]
            }]
        )
        return response.output_text
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
    
async def vlm_process(image_path: str, task_type: str) -> str:
    try:
        prompt = prepare_prompt("process", task_type)
        message = {'role': 'user', 
                   'content': '请详细描述图片内容',
                   'images':{image_path}
                   }
        response = await AsyncClient().chat(model='qwen2.5vl:7b-fp16', messages=[message])
        return response.message.content
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
