import os
import json
import logging 
from pathlib import Path

from LLM.LLMclient import ChatModel
from PromptTune.answer_generation import UNIMODAL_GENERATE_ANSWER_PROMPT, MULTIMODAL_GENERATE_ANSWER_PROMPT, VLM_GENERATE_ANSWER_PROMPT

def _chunk_organizer(chunk_data: dict, 
                     image_file_dir: str = None
                     ) -> tuple[list, list, list]:
    """
    解包 chunks 数据，分别存放 text chunk 和 multimodal chunk
    对 text chunk 提取其中的文本，对 multimodal chunk 提取其中的文本和图片路径

    参数:
    chunk_data: {chunkID: chunk_dict}
    image_file_dir: 图片文件的文件夹路径

    返回:
    text_list: [text, ...]
    mm_text_list: [text,...]
    image_list: [image_path,...]
    """
    text_data = chunk_data.get("text", {})
    multimodal_data = chunk_data.get("multimodal", {})

    text_list = []
    mm_text_list = []
    image_list = []
    for key, value in text_data.items():
        if value.get("type") == "text":
            text = [value.get('text', '')]
            text_list.extend(text)
    if image_file_dir is not None:
        for key, value in multimodal_data.items():
            if value.get("type") == "image":
                img_path = value.get('img_path', '')
                img_caption = value.get('image_caption', [])
                img_footnote = value.get('image_footnote', [])

                image_path = [os.path.join(image_file_dir, img_path)]
                image_list.extend(image_path)
                mm_text_list.extend(img_caption)
                mm_text_list.extend(img_footnote)
            elif value.get("type") == "equation":
                img_path = value.get('img_path', '')
                text = value.get('text', '')

                image_path = [os.path.join(image_file_dir, img_path)]
                image_list.extend(image_path)
                mm_text_list.extend([text])
            elif value.get("type") == "table":
                img_path = value.get('img_path', '')
                table_caption = value.get('table_caption', '')
                table_footnote = value.get('table_footnote', '')
                table_body = value.get('table_body', [])

                image_path = [os.path.join(image_file_dir, img_path)]
                image_list.extend(image_path)
                mm_text_list.extend(table_caption)
                mm_text_list.extend(table_footnote)
                mm_text_list.extend([table_body])
    
    return text_list, mm_text_list, image_list

def unimodal_generator(query: str, 
                       answer: str, 
                       chunks: dict, 
                       chatLLM: ChatModel, 
                       save_dir: str = None
                       ) -> str:
    """
    生成回答
    输入:
        query: 用户问题
        answer: 正确答案
        chunks: {chunkID: chunk_dict}
        chatLLM: ChatModel
        save_dir: 保存文件夹名
    返回:
        response: 回答
    """
    text_list, _, _ = _chunk_organizer(chunk_data=chunks, image_file_dir=None) 
    docs_str = " ".join(text_list) if isinstance(text_list, list) else text_list

    prompt = UNIMODAL_GENERATE_ANSWER_PROMPT.format(query=query, text_chunks=docs_str)
    response = chatLLM.chat_with_llm(prompt=prompt, 
                                     system_content="You are a AI assistant tasked with answering questions based on provided textual information.")
    logging.info(f"单模态生成答案 {response}")
    
    # 如果指定了保存路径，则尝试保存
    if save_dir:
        try:
            # 自动创建目录
            save_path = os.path.join(save_dir, query + ".json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {"query": query, 
                    "prompt": prompt,
                    "response": response, 
                    "answer": answer}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"单模态生成答案已保存为 {save_path}")
            logging.info(f"单模态生成答案已保存为 {save_path}")
        except Exception as e:
            print(f"单模态生成答案保存失败：{e}")
            logging.error(f"单模态生成答案保存失败：{e}")

    return prompt, response

def multimodal_generator(query: str, 
                         answer: str,
                         chunks: dict, 
                         chatVLM: ChatModel, 
                         ocr_imagefile_dir: str, 
                         save_dir: str = None
                         ) -> str:
    """
    生成回答
    输入:
        query: 用户问题
        answer: 正确答案
        chunks: {chunkID: chunk_dict}
        chatVLM: ChatModel
        ocr_imagefile_dir: 图片文件的文件夹路径
        save_dir: 保存文件夹名
    返回:
        response: 回答
    """
    text_list, mm_text_list, image_list = _chunk_organizer(chunk_data=chunks, 
                                                           image_file_dir=Path(ocr_imagefile_dir)) 
    docs_str = " ".join(text_list) if isinstance(text_list, list) else text_list
    mm_docs_str = " ".join(str(x) for x in mm_text_list) if isinstance(mm_text_list, list) else str(mm_text_list)
    
    prompt = MULTIMODAL_GENERATE_ANSWER_PROMPT.format(query=query, 
                                                      text_chunks=docs_str, 
                                                      image_description=mm_docs_str)
    response = chatVLM.chat_with_mullm(image_paths=image_list, 
                                       prompt=prompt, 
                                       system_content="You are a AI assistant tasked with answering questions based on provided textual and visual information.")
    logging.info(f"多模态生成答案 {response}")
    
    # 如果指定了保存路径，则尝试保存
    if save_dir:
        try:
            # 自动创建目录
            save_path = os.path.join(save_dir, query + ".json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {"query": query, 
                    "prompt": prompt,
                    "image": image_list,
                    "response": response, 
                    "answer": answer}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"多模态生成答案已保存为 {save_path}")
            logging.info(f"多模态生成答案已保存为 {save_path}")
        except Exception as e:
            print(f"多模态生成答案保存失败：{e}")
            logging.error(f"多模态生成答案保存失败：{e}")
    
    return prompt, image_list, response

def vlm_generator(query: str, 
                         answer: str,
                         chatVLM: ChatModel, 
                         image_list: str, 
                         save_dir: str = None
                         ) -> str:
    """
    生成回答
    输入:
        query: 用户问题
        answer: 正确答案
        chatVLM: ChatModel
        save_dir: 保存文件夹名
    返回:
        response: 回答
    """
    prompt = VLM_GENERATE_ANSWER_PROMPT.format(query=query)
    response = chatVLM.chat_with_mullm(image_paths=image_list, 
                                       prompt=prompt, 
                                       system_content="You are a AI assistant tasked with answering questions based on provided textual and visual information.")
    logging.info(f"多模态生成答案 {response}")
    
    # 如果指定了保存路径，则尝试保存
    if save_dir:
        try:
            # 自动创建目录
            save_path = os.path.join(save_dir, query + ".json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {"query": query, 
                    "prompt": prompt,
                    "image": image_list,
                    "response": response, 
                    "answer": answer}
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"多模态生成答案已保存为 {save_path}")
            logging.info(f"多模态生成答案已保存为 {save_path}")
        except Exception as e:
            print(f"多模态生成答案保存失败：{e}")
            logging.error(f"多模态生成答案保存失败：{e}")
    
    return prompt, image_list, response