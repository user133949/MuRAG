import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from LLM.LLMclient import ChatModel
from PromptTune.image_description import (IMAGE_ANALYSIS_SYSTEM, 
                                          TABLE_ANALYSIS_SYSTEM, 
                                          EQUATION_ANALYSIS_SYSTEM, 
                                          IMAGE_ANALYSIS_PROMPTS, 
                                          TABLE_ANALYSIS_PROMPTS, 
                                          EQUATION_ANALYSIS_PROMPTS)
from PromptTune.paragraph_summarization import (TEXT_SUMMARY_SYSTEM, 
                                              TEXT_SUMMARY_PROMPTS)

def create_id_and_embedding(extraction_result: str, 
                            page_idx: int, 
                            segment_idx: int, 
                            pdf_name: str,
                            encoder: SentenceTransformer
                            ) -> list:
    """
    将 extraction_result 字符串转换为列表对象，并为每个实体添加 entityID 属性
    以及计算 embedding。
    
    参数:
        extraction_result (str): JSON 格式的字符串，表示三元组列表。
        page_idx (int): 页码索引。
        segment_idx (int): 段落索引。
        pdf_name (str): 参考文件的唯一名称。
        encoder (SentenceTransformer): 用于获取文本embedding的编码器。
        
    返回:
        list: 添加了 entityID 的字典列表。
    """
    try:
        triplet = json.loads(extraction_result)  # 字符串转换为列表对象
    except json.JSONDecodeError:
        print("警告: LLM返回的不是有效的 JSON 格式，跳过此段落。")
        return []
    
    for idx, item in enumerate(triplet):
        # 对节点进行处理
        if isinstance(item, dict) and all(k in item for k in ("name", "type", "description")):
            item["entityID"] = str([page_idx, segment_idx, idx])
            item["chunkID"] = str([[page_idx, segment_idx]])
            item["ref_doc_id"] = pdf_name
            # 将名称和描述组合，格式化成 "名称: 描述"，计算 embedding
            entity_name = item.get("name", "")
            entity_description = item.get("description", "")
            text_to_embed = f"{entity_name}: {entity_description}"
            item["vector"] = encoder.encode(text_to_embed).tolist()
        # 对边进行处理
        elif isinstance(item, dict) and all(k in item for k in ("source", "target", "relationship")):
            item["ref_doc_id"] = pdf_name
    return triplet

def create_anchor_node(pdf_name: str,
                       segment_text: str | list[str], 
                       page_idx: int, 
                       segment_idx: int, 
                       chatLLM: ChatModel, 
                       encoder: SentenceTransformer
                       ) -> list:
    """
    创建一个段落的锚节点的字典列表。

    参数:
    - pdf_name (str): 参考文件的唯一名称
    - segment_text (str): 段落文本内容
    - page_idx (int): 页面索引
    - segment_idx (int): 段落索引
    - chatLLM (ChatModel): 用于生成锚节点描述的LLM模型

    返回:
    - list[dict]: 包含锚节点的list
    """
    system = TEXT_SUMMARY_SYSTEM
    prompt = TEXT_SUMMARY_PROMPTS

    if segment_text is not None:
        prompt = prompt.format(page_idx=page_idx, 
                               segment_idx=segment_idx, 
                               text=segment_text)
    segment_description = chatLLM.chat_with_llm(prompt=prompt, 
                                                system_content=system)
    if segment_description is not None:
        description = segment_description
    else:
        description = "Anchor node used to index contextual paragraphs"

    anchor_node = {
        "name": f"segment anchor node for page {page_idx}, segment {segment_idx}",
        "type": "SEGMENT ANCHOR NODE",
        "description": description,
        "entityID": str([page_idx, segment_idx, "anchor"]), 
        "chunkID": str([[page_idx, segment_idx]]), 
        "ref_doc_id": pdf_name
    }
    # 使用段落的 description 来为anchor_node本身生成向量
    anchor_node["vector"] = encoder.encode(description).tolist()

    return [anchor_node]

def create_anchor_edge(pdf_name, 
                       source_triplet: List[Dict], 
                       target_triplet: List[Dict], 
                       local: bool | None=True,
                       strength: int | None=3
                       ) -> List[Dict]:
    """
    创建指定三元组与目标锚节点的双向连边，默认强度为 3。

    参数:
        pdf_name: 参考文件的唯一名称
        source_triplet: 当前段落的三元组列表
        target_triplet: 目标三元组（锚节点）
        local: 指定边的类型是本段还是上一段
        strength: 关联强度，默认强度为 3
    """
    anchor_edges = []
    source_nodes = [item["name"] for item in source_triplet if "name" in item]
    target_nodes = [item["name"] for item in target_triplet if "name" in item]
    
    if local:
        for a in source_nodes:
            for b in target_nodes:
                anchor_edges.append({
                    "source": a,
                    "target": b,
                    "relationship": "edges to local segment anchor node",
                    "relationship_strength": strength, 
                    "ref_doc_id": pdf_name
                })
                anchor_edges.append({
                    "source": b,
                    "target": a,
                    "relationship": "edges to local segment anchor node",
                    "relationship_strength": strength, 
                    "ref_doc_id": pdf_name
                })
    else:
        for a in source_nodes:
            for b in target_nodes:
                anchor_edges.append({
                    "source": a,
                    "target": b,
                    "relationship": "edges to last segment anchor node",
                    "relationship_strength": strength, 
                    "ref_doc_id": pdf_name
                })
                anchor_edges.append({
                    "source": b,
                    "target": a,
                    "relationship": "edges to last segment anchor node",
                    "relationship_strength": strength, 
                    "ref_doc_id": pdf_name
                })
    
    return anchor_edges

def creat_image_node(chatVLM: ChatModel, 
                     pdf_name: str,
                     imagefile_dir: str, 
                     img_path: str, 
                     node_type: str, 
                     docs: str | list[str], 
                     page_idx: int, 
                     segment_idx: int, 
                     encoder: SentenceTransformer
                     ) -> list:
    """
    为图片创建原始图片节点
    
    参数:
        chatVLM: 视觉大语言模型实例
        pdf_name (str): 参考文件的唯一名称
        imagefile_dir (str): 图片文件目录路径
        img_path (str): 图片路径
        node_type(str): 节点类型，('image', 'equation', 'table')
        docs(str/list[str]): 图片相关的文本内容
        page_idx (int): 页码索引
        segment_idx (int): 段落索引
        encoder (SentenceTransformer): 用于获取文本embedding的编码器。
    返回:
        list: 图片节点的字典列表
    """
    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    # 先根据类型，提供提示词模板
    if node_type == "image":
        system = IMAGE_ANALYSIS_SYSTEM
        prompt = IMAGE_ANALYSIS_PROMPTS
    elif node_type == "equation":
        system = EQUATION_ANALYSIS_SYSTEM
        prompt = EQUATION_ANALYSIS_PROMPTS
    elif node_type == "table":
        system = TABLE_ANALYSIS_SYSTEM
        prompt = TABLE_ANALYSIS_PROMPTS

    if docs is not None:
        prompt = prompt.format(context=docs_str)

    image_path = os.path.join(imagefile_dir, img_path)
    image_description = chatVLM.chat_with_mullm(image_paths=[image_path], 
                                                prompt=prompt, 
                                                system_content=system)

    description_text = f"{img_path} || {image_description}" 
    image_node = {
        "name": f"image in page {page_idx} and segment {segment_idx}", 
        "type": "ORIGINAL_IMAGE", 
        "description": description_text,
        "entityID": str([page_idx, segment_idx, "image"]), 
        "chunkID": str([[page_idx, segment_idx]]), 
        "ref_doc_id": pdf_name
    }
    # 使用图片 路径+上下文 来为图片节点本身生成向量
    image_node["vector"] = encoder.encode(description_text).tolist()

    return [image_node], image_description

def creat_image_edges(pdf_name: str, 
                      source_triplet: List[Dict], 
                      target_triplet: List[Dict], 
                      strength: int | None=3
                      ) -> List[Dict]:
    """
    为图片创建原始图片与段落三元组的连边，默认强度为 3。

    参数:
        pdf_name (str): 参考文件的唯一名称
        source_triplet: 原图节点的字典列表
        target_triplet: 图片三元组(caption、footnote)
        strength: 关联强度，默认强度为 3
    
    返回:
        list: deges的字典列表
    """
    image_edges = []
    source_nodes = [item["name"] for item in source_triplet if "name" in item]
    target_nodes = [item["name"] for item in target_triplet if "name" in item]
    
    for a in source_nodes:
        for b in target_nodes:
            image_edges.append({
                "source": a,
                "target": b,
                "relationship": "the origin image of triplet",
                "relationship_strength": strength, 
                "ref_doc_id": pdf_name
            })
            image_edges.append({
                "source": b,
                "target": a,
                "relationship": "the origin image of triplet",
                "relationship_strength": strength, 
                "ref_doc_id": pdf_name
            })
    
    return image_edges

def creat_equation_node(pdf_name: str, 
                        equation_text: str, 
                        page_idx: int, 
                        segment_idx: int, 
                        encoder: SentenceTransformer
                        ) -> list:
    """
    为公式创建公式内容节点
    
    参数:
        pdf_name (str): 参考文件的唯一名称
        equation_text (str): 公式文本内容
        page_idx (int): 页码索引
        segment_idx (int): 段落索引
        encoder (SentenceTransformer): 用于获取文本embedding的编码器。
        
    返回:
        list: 公式内容节点的字典列表
    """
    equation_node = {
        "name": f"equation node for page {page_idx}, segment {segment_idx}", 
        "type": "EQUATION NODE", 
        "description": equation_text, 
        "entityID": str([page_idx, segment_idx, "equation"]), 
        "chunkID": str([[page_idx, segment_idx]]), 
        "ref_doc_id": pdf_name
        }
    equation_node["vector"] = encoder.encode(equation_text).tolist()

    return [equation_node]

def creat_table_node(pdf_name: str, 
                     table_body: str, 
                     page_idx: int, 
                     segment_idx: int, 
                     encoder: SentenceTransformer
                     ) -> list:
    """
    为表格创建表格内容节点
    
    参数:
        pdf_name (str): 参考文件的唯一名称
        table_body (str): 表格文本内容
        page_idx (int): 页码索引
        segment_idx (int): 段落索引
        encoder (SentenceTransformer): 用于获取文本embedding的编码器。
        
    返回:
        list: 表格内容节点的字典列表
    """
    table_node = {
        "name": f"table node for page {page_idx}, segment {segment_idx}", 
        "type": "TABLE NODE", 
        "description": table_body, 
        "entityID": str([page_idx, segment_idx, "table"]), 
        "chunkID": str([[page_idx, segment_idx]]), 
        "ref_doc_id": pdf_name
        }
    table_node["vector"] = encoder.encode(table_body).tolist()

    return [table_node]