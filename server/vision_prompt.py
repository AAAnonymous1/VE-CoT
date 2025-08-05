"""
format prompt for vlm describe
"""
ROUGH_DESCRIPTION_DATASET_PROMPT="""
## Task Description:
    The current task is to briefly describe a remote-sensing image; your part of the task is:
    1. Upon receiving a satellite remote-sensing image, you must provide an overall, simple description. Make sure the description does **not** contain fine-grained identifying information, but category-level descriptions are allowed.
    2. Do **not** output any irrelevant content; our entire dialogue will be used as the dataset.
    3. Do **not** use code blocks and do not output acknowledgements such as “received”.
    4. When writing the VLM description, do **not** give next-step suggestions like chat; focus only on describing the image.
    5. The task may involve multiple rounds of dialogue; strictly follow these requirements in every round.
    6. The description example is for reference only; describe according to the actual content.
## Description Example:
    This is a high-resolution satellite image showing an aircraft carrier berthed at a harbour. Numerous carrier-borne aircraft are neatly parked on the deck. A white number is present at the stern, but the resolution is insufficient to distinguish the exact digits. Roads, buildings, and vehicles are laid out regularly around the port area.
"""

TOOL_OUTPUT_DESCRIBE_DATASET_PROMPT="""
## Task Description:
    The current task is to give a detailed description of the tool-processed image; your part of the task is:
    1. Given the remote-sensing image after tool processing, provide a detailed description strictly according to the image content.
    2. Do **not** use code blocks and do not output acknowledgements such as “received”.
    3. When writing the VLM description, do **not** give next-step suggestions like chat; focus only on describing the image.
    4. The task may involve multiple rounds of dialogue; strictly follow these requirements in every round.
    5. If the image contains bounding boxes, describe each boxed region in detail.
    6. The description example is for reference only; describe according to the actual content.
    7. Provide as much fine-grained description as possible to facilitate subsequent information extraction.
## Description Example:
    When the image has no boxes, simply describe the image content. If boxes are present, describe each box separately, including its coordinates and content; the box data come from GroundingDINO output.
    Bounding-box description example:
        Upper-left water area: a box selects a ship, category “cargo ship”, confidence 0.393, coordinates (3, 162, 1245, 653), covering the entire main subject, likely a mis-classification.
        Main ship area: a large box labelled “military warship cargo ship”, confidence 0.379, coordinates (129, 67, 1042, 409), covering the entire aircraft-carrier structure.
    Regular image description example (binarisation case):
        The binarised, enhanced image shows {{detailed description of the image}}
"""

ROUGH_DESCRIPTION_PROCESS_PROMPT="""
"""

TOOL_OUTPUT_DESCRIBE_PROCESS_PROMPT="""
"""

def prepare_prompt(procedure_type,task_type):
    """
    parsing prompt requirements and return specific prompt for vlm
    Args:
        procedure_type:distinguish current process is dataset format(dataset) or process(process)
        task_type:distinguish current visual task is rough discription(rough) or tool detailed describe(tool)
    """
    if procedure_type == "dataset":
        if task_type == "rough":
            return ROUGH_DESCRIPTION_DATASET_PROMPT
        elif task_type == "tool":
            return TOOL_OUTPUT_DESCRIBE_DATASET_PROMPT
        else:
            raise ValueError(f"task type: {task_type} unrecognized")
    elif procedure_type == "process":
        if task_type == "rough":
            return ROUGH_DESCRIPTION_PROCESS_PROMPT
        elif task_type == "tool":
            return TOOL_OUTPUT_DESCRIBE_PROCESS_PROMPT
        else:
            raise ValueError(f"task type: {task_type} unrecognized")
    else:
        raise ValueError(f"procedure type: {procedure_type} unrecognized")
    
if __name__ == "__main__":
    print(prepare_prompt("dataset","tool"))