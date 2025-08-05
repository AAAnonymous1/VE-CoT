"""
mcp vision tools server for dataset
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Real_ESRGAN")))
from mcp.server.fastmcp import FastMCP
import mcp.types as types
import cv2
from pathlib import Path
from gdino_run import GDINO_CONFIG, gdino_run
from utils import vlm_prepare_dataset, vlm_process
from Real_ESRGAN.real_esrgan_run import REAL_ESRGAN_CONFIG, real_esrgan_run
from Restormer.restormer_run import RESTORMER_CONFIG, restomer_run
import asyncio
from datetime import datetime

vision_server = FastMCP("opencv tools")

def get_new_path(image_path: str, suffix: str, ext: str=None):
    """
    format temp path for intermediate result image
    Args:
        image_path:origin image path
        suffix:new image name suffix, represents process type
        ext:some process may need change extension
    """
    origin_path = Path(image_path)
    filename = origin_path.name
    stem = origin_path.stem
    if not ext:
        ext = origin_path.suffix
    parentdir = origin_path.parent

    now = datetime.now()
    timestamp = now.strftime("%y%m%d%H%M%S")

    if "temp" not in parentdir.parts:
        os.makedirs(f"{parentdir}/temp", exist_ok=True)
        new_path = f"{parentdir}/temp/{stem}_{suffix}_{timestamp}{ext}"
    else:
        new_path = f"{parentdir}/{stem}_{suffix}_{timestamp}{ext}"
    return new_path


@vision_server.tool()
async def image_gray(image_path: str):
    """
    Grayscale conversion of the image using OpenCV
    Args:
        image_path:input image path
    """
    try:
        img = cv2.imread(image_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_image_path = get_new_path(image_path, "gray")
        cv2.imwrite(new_image_path, gray_image)
        response = await vlm_prepare_dataset(new_image_path, 'tool')
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
    return {"result_image_path":new_image_path, "vlm_response":response}

@vision_server.tool()
async def image_binary(image_path: str):
    """Binarization using OpenCV enhances the clarity of numerical and patterned regions in the image.
    Args:
        image_path:input image path
    """
    try:
        img = cv2.imread(image_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(
            gray_image, 
            thresh=127,
            maxval=255,
            type=cv2.THRESH_TOZERO
        )
        new_image_path = get_new_path(image_path, "binary")
        cv2.imwrite(new_image_path, binary_image)
        response = await vlm_prepare_dataset(new_image_path, 'tool')
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
    return {"result_image_path":new_image_path,"vlm_response":response}

@vision_server.tool()
async def image_crop(image_path:str, x1:int, y1:int, x2:int, y2:int):
    """
    Crop the rectangular region of the image using OpenCV.
    Args：
        image_path:input image path
        x1:Horizontal coordinates of the point in the upper left corner
        y1:Vertical coordinates of the point in the upper left corner
        x2:Horizontal coordinates of the lower right point
        y2:Vertical coordinate of the lower right point
    """
    try:
        image = cv2.imread(image_path)

        cropped = image[y1:y2, x1:x2]

        new_image_path = get_new_path(image_path, "cropped")
        cv2.imwrite(get_new_path(image_path, "cropped"), cropped)
        response = await vlm_prepare_dataset(new_image_path, 'tool')
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
    return {"result_image_path":new_image_path, "vlm_response":response}

@vision_server.tool()
async def image_detection(image_path:str, txt_prompt:str):
    """
    Invoke GroundingDINO to detect image content based on the input text prompt.
    Args:
        image_path:input image text
        txt_prompt:Enter the prompts for GroundingDINO to detect. Focus on key targets using nouns or adjective+noun phrases, and separate each item with a period. For example: military warship . cargo ship . aircraft . warship tail number . aircraft tail number . marking on tail of ship . marking of aircraft . building on port
    """
    try:
        origin_path = Path(image_path)
        annotated_image_path = f"{origin_path.parent}/temp/{origin_path.stem}_annotated.png"
        annotated_txt_path = f"{origin_path.parent}/temp/{origin_path.stem}_boxes.txt"
        with open(annotated_txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        response = await vlm_prepare_dataset(annotated_image_path, 'tool', dino=lines)
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
    return {"result_image_path":annotated_image_path, "boxes":lines, "vlm_response":response}

@vision_server.tool()
async def vlm_describe(image_path: str, text_prompt: str):
    """
    Invoke the VLM model to detect image content based on the input text prompts.
    Args:
        image_path:input image path
        text_prompt:Cue words to focus on
    """
    try:
        if text_prompt == "rough":
            response = await vlm_prepare_dataset(image_path, 'rough')
        else:
            response = await vlm_prepare_dataset(image_path, 'tool', text_prompt=text_prompt)
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
    return {"vlm_response":response}


@vision_server.tool()
async def image_super_resolution(image_path:str):
    """
    Apply the Real-ESRGAN tool to perform super-resolution processing on the blurred image obtained from the cropped images.
    Args:
        image_path:input image path
    """
    try:
        new_image_path = get_new_path(image_path,"esrgan")
        REAL_ESRGAN_CONFIG.output_dir = new_image_path
        REAL_ESRGAN_CONFIG.input_dir = image_path
        real_esrgan_run(REAL_ESRGAN_CONFIG)
        response = await vlm_prepare_dataset(new_image_path, 'tool')
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
    return {"result_image_path":new_image_path, "vlm_response":response}

@vision_server.tool()
async def image_restormer(image_path:str, task_type:str="Motion_Deblurring"):
    """
    Invoke the Restormer tool to perform Motion Blur Removal, Defocus Blur Removal, Deraining, Real Noise Removal, and Gaussian Color Noise Removal on the image.
    Args:
        image_path:input image path
        task_type:task type，please select one of the following:Motion_Deblurring | Single_Image_Defocus_Deblurring | Deraining | Real_Denoising | Gaussian_Color_Denoising
    """
    try:
        new_image_path = get_new_path(image_path,"restormer")
        RESTORMER_CONFIG.input_dir = image_path
        RESTORMER_CONFIG.output_dir = new_image_path
        RESTORMER_CONFIG.task = task_type
        restomer_run(RESTORMER_CONFIG)
        response = await vlm_prepare_dataset(new_image_path, 'tool')
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
    return {"result_image_path":new_image_path, "vlm_response":response}


if __name__=="__main__":
    vision_server.run(transport="stdio")