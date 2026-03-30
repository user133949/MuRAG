import re
import json
import logging
from LLM.LLMclient import ChatModel

def _is_valid_item(item: dict) -> bool:
    """
    校验每个条目是否符合实体或关系的规范。
    """
    entity_keys = {"name", "type", "description"}
    relation_keys = {"source", "target", "relationship", "relationship_strength"}

    if not isinstance(item, dict):
        return False
    if entity_keys.issubset(item.keys()):
        return True
    if relation_keys.issubset(item.keys()):
        return True
    return False

def entity_relationship_extraction(chat_model: ChatModel, 
                                   persona: str, 
                                   extract_prompt: str, 
                                   max_retries=3
                                   ) -> str:
    """
    生成 json 格式的实体关系提取响应，
    并通过保存 json 文件检查生成的格式是否正确

    参数:
        chat_model (ChatModel): 用于生成响应的聊天模型
        persona (str): 系统角色描述
        extract_prompt (str): 提取实体关系的提示
        max_retries (int, optional): 最大重试次数.

    返回:
        str: 生成的 json 字符串，如果保存失败则返回空字符串
    """
    for attempt in range(max_retries):
        response = chat_model.chat_with_llm(prompt=extract_prompt, 
                                            system_content=persona)
        try:
            # 用正则提取第一个有效的 JSON
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if not match:
                logging.error(f"第{attempt+1}次提取 JSON 失败，正在重试...")
                continue
            # 检查 JSON 格式是否合法
            json_str = match.group(0)
            data = json.loads(json_str)
            # 检查每个条目是否符合实体或关系的规范
            if all(_is_valid_item(entry) for entry in data):
                return json_str     # 校验通过才返回
        except Exception as e:
            logging.error(f"第{attempt+1}次提取 JSON 失败：{e}，正在重试...")
    logging.error(f"尝试 {max_retries} 次后仍未成功，已放弃。")

    return ""  # 失败时返回空字符串