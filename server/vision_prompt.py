"""
format prompt for vlm describe
"""
ROUGH_DESCRIPTION_DATASET_PROMPT="""
## 任务说明：
    当前任务为简要描述遥感图像，你需要完成的任务部分为：
    1. 输入遥感卫星图像，你需要对图像进行总体性的简单描述，请注意当前描述不要出现详细的可用于判别的具体信息，但可以出现类别描述
    2. 请你不要输出任何无关内容，我们的对话内容将全部作为数据集
    3. 请不要用代码框输出，也不要输出收到
    4. VLM描述时请不要像chat那样给出下一步建议，专注于图像描述即可
    5. 当前任务可能涉及多轮对话，请每轮都严格遵循要求
    6. 描述示例仅供参考，请根据具体内容进行描述
## 描述示例：
    这是一张高分辨率卫星图像，显示一艘航空母舰停靠在港口。甲板上停放有大量舰载机，呈规则排列。舰艉位置有白色编号，但清晰度不足，难以辨认具体数字。周边港区有道路、建筑与车辆，环境布局规整。
"""
TOOL_OUTPUT_DESCRIBE_DATASET_PROMPT="""
## 任务说明：
    当前任务为详细描述工具处理结果图像，你需要完成的任务部分为：
    1. 输入遥感图像经工具处理后的结果图像，你需要对结果图像进行详细的描述，同时请严格按照图像内容描述
    2. 请不要用代码框输出，也不要输出收到
    3. VLM描述时请不要像chat那样给出下一步建议，专注于图像描述即可
    4. 当前任务可能涉及多轮对话，请每轮都严格遵循要求
    5. 如果是框图请针对每个框图区域进行细致描述
    6. 描述示例仅供参考，请根据具体内容进行描述
    7. 请尽可能多地进行细粒度描述，以帮助后续信息提取
## 描述示例：
    VLM描述时如果没有框则直接描述图像内容，如果图中有框请分别对每个框进行描述，需包含框的坐标以及内容，框图描述会输入GroundingDINO输出的框及其坐标
    框图描述示例：
        左上水面区域框选出一艘船只，类别为“cargo ship”，置信度0.393，坐标为(3, 162, 1245, 653)，覆盖整个图像主体，疑似误判。
        主体舰船区域框出大范围区域，类别为“military warship cargo ship”，置信度0.379，坐标为(129, 67, 1042, 409)，覆盖航空母舰全体结构。
    常规图像描述示例（以二值化处理为例）：
        二值化增强后的图像，显示{{详细描述图像}}
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