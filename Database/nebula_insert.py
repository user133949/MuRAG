"""
ingest/nebula_ingest.py
把 JSON 图数据（单文件或目录）批量写入 Nebula。
提供函数：ingest_graph_data(...)
"""

import os
import json
from typing import Optional, Dict
try:
    # 当作为模块导入时
    from .joint import JointHandler
except ImportError:
    from joint import JointHandler

def ingest_graph_data(
    space: str,
    input_path: str,
    nebula_host: str = "127.0.0.1",
    nebula_port: int = 9669,
    embedding_dim: int = 1024,
    entity_batch: int = 400,
    relation_batch: int = 400,
    reset: bool = True,
) -> Dict:
    """
    将 JSON 文件或目录内的多个 JSON 文件加载并写入 Nebula（通过 JointHandler）。

    Args:
        space: Nebula space 名称（原来的 SPACE）。
        input_path: 单个 JSON 文件路径或包含 JSON 的目录路径。
        nebula_host: Nebula 地址，默认本机。
        nebula_port: Nebula 端口，默认 9669。
        embedding_dim: 嵌入向量维度，传给 joint.setup。
        entity_batch: 批量写入的实体大小。
        relation_batch: 批量写入的关系大小。
        reset: 是否在写入前重置（drop & create）space。

    Returns:
        dict: 简单汇总信息，例如处理的文件数与元素总数。
    """
    joint = JointHandler(space_name=space, nebula_host=nebula_host, nebula_port=nebula_port)
    joint.setup(embedding_dim=embedding_dim)

    processed_files = 0
    total_elements = 0
    try:
        if reset:
            # 如果需要重置 space，则调用 nebula 的重置方法
            try:
                joint.nebula.reset_space()
            except Exception as e:
                # 不要阻塞，记录异常
                print(f"警告：重置 space 失败：{e}")

        # 目录情况：按原脚本只匹配以 final_graph.json 结尾的文件
        if os.path.isdir(input_path):
            json_files = [f for f in os.listdir(input_path) if f.endswith('final_graph.json')]
            print(f"找到 {len(json_files)} 个 JSON 文件，将它们的图数据插入数据库...")
            for json_file in json_files:
                file_path = os.path.join(input_path, json_file)
                print(f"正在处理文件: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    final_graph_with_vectors = json.load(f)
                print(f"--- 知识图谱载入，共包含 {len(final_graph_with_vectors)} 个元素 ---")

                # 清理换行
                for item in final_graph_with_vectors:
                    if 'description' in item and isinstance(item['description'], str):
                        item['description'] = item['description'].replace('\n', ' ')

                joint.ingest_graph_data_bulk(
                    final_graph_with_vectors,
                    entity_batch=entity_batch,
                    relation_batch=relation_batch,
                    reset=False
                )
                processed_files += 1
                total_elements += len(final_graph_with_vectors)

        # 单文件情况
        elif os.path.isfile(input_path) and input_path.endswith('.json'):
            print(f"正在处理文件: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                final_graph_with_vectors = json.load(f)
            print(f"--- 知识图谱载入，共包含 {len(final_graph_with_vectors)} 个元素 ---")
            for item in final_graph_with_vectors:
                if 'description' in item and isinstance(item['description'], str):
                    item['description'] = item['description'].replace('\n', ' ')
            # 如果是单文件，保留 reset 参数语义：这里以 reset 参数决定是否 reset space（外部已处理）
            joint.ingest_graph_data_bulk(
                final_graph_with_vectors,
                entity_batch=entity_batch if entity_batch else 500,
                relation_batch=relation_batch if relation_batch else 800,
                reset=reset
            )
            processed_files = 1
            total_elements = len(final_graph_with_vectors)
        else:
            msg = f"输入路径 {input_path} 无效。请提供有效的 JSON 文件或包含 final_graph.json 的文件夹。"
            print(msg)
            return {"success": False, "message": msg}

        return {
            "success": True,
            "processed_files": processed_files,
            "total_elements": total_elements,
            "space": space
        }
    finally:
        # 确保关闭连接
        try:
            joint.close()
        except Exception:
            pass
