import re
import json
import logging
from typing import Dict
from PromptTune.rag_system import SYSTEM_PARAMETER
from LLM.LLMclient import ChatModel

def get_system_parameter(model: str, 
                         reasoning_model: str, 
                         query: str,  
                         api_key="ollama", 
                         base_url="http://localhost:11434/v1/"
                         ) -> str:
    """
    调用LLM获取系统参数。
    """
    chatLLM = ChatModel(model=model, 
                        reasoning_model=reasoning_model,
                        api_key=api_key, 
                        base_url=base_url)
    try:
        response = chatLLM.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": SYSTEM_PARAMETER.format(input_text=query),
                }
            ],
            temperature=0.0,
            max_tokens=256
        )
        response = response.choices[0].message.content
    except:
        logging.error("Failed to get rag system parameter")
        response = "Failed"
    
    return response

def extract_weights(llm_response: str, default: float = 0.5) -> Dict[str, float]:
    """
    从LLM响应中提取权重参数。
    参数:
        llm_response (str): LLM响应字符串
        default (float): 默认权重值
    """
    # 默认权重值
    allowed_keys = {"alpha", "beta", "lam"}
    weights = {key: default for key in allowed_keys}

    try:
        fixed_str = re.sub(r",\s*}", "}", llm_response)
        fixed_str = re.sub(r",\s*]", "]", fixed_str)
        response_json = json.loads(fixed_str)
    except Exception:
        logging.error("SystemParameter - Failed to parse json")
        response_json = None

    if response_json:
        # 检查数值是否在0到1之间
        for key in weights.keys():
            value = response_json.get(key)
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                weights[key] = float(value)
            else:
                weights[key] = default
        
    # 如果JSON解析失败，使用默认值
    return weights
