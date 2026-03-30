import os
import json
import logging
from PromptTune.answer_extraction import ANSWER_EXTRACTION_PROMPT
from LLM.LLMclient import ChatModel

def extract_answer(model: str, 
                   reasoning_model: str,
                   question: str, 
                   output: str, 
                   save_dir: str = None,
                   api_key="ollama", 
                   base_url="http://localhost:11434/v1/"
                   ):
    """从模型回复中提取关键信息"""
    chatLLM = ChatModel(model=model, 
                        reasoning_model=reasoning_model,
                        api_key=api_key, 
                        base_url=base_url)
    try:
        response = chatLLM.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_EXTRACTION_PROMPT + "/no_think",
                },
                {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
                }
            ],
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = response.choices[0].message.content
    except:
        response = "Failed"
    
    # 如果指定了保存路径，则尝试保存
    if save_dir:
        try:
            # 自动创建目录
            save_path = os.path.join(save_dir, question + "_extract.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {"query": question, 
                    "output": output, 
                    "extract": response}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"多模态生成答案已保存为 {save_path}")
            logging.info(f"多模态生成答案 {save_path}")
        except Exception as e:
            print(f"多模态生成答案保存失败：{e}")
            logging.error(f"多模态生成答案保存失败：{e}")
    
    return response

