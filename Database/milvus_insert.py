"""
ingest/milvus_ingest.py
把 *_final_graph_with_vector.json 中的向量/文本数据写入 Milvus（通过 JointHandler）。
提供函数：ingest_vector_data(...)
"""

import os
import json
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer
try:
    # 当作为模块导入时
    from .joint import JointHandler
except ImportError:
    from joint import JointHandler

def ingest_vector_data(
    space: str,
    input_path: str,
    nebula_host: str = "127.0.0.1",
    nebula_port: int = 9669,
    milvus_host: str = "127.0.0.1",
    milvus_port: int = 19530,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = "Qwen/Qwen3-Embedding-0.6B",
    reset_collection: bool = True,
) -> Dict:
    """
    将 JSON 向量文件批量写入 Milvus（通过 JointHandler）。

    Args:
        space: 逻辑空间名（也可作为 collection 名）。
        input_path: 单个 JSON 文件或目录路径，会匹配 *_final_graph_with_vector.json。
        nebula_host / nebula_port: Nebula 主机与端口（JointHandler 可能需要）。
        milvus_host / milvus_port: Milvus 主机与端口。
        collection_name: Milvus collection 名；若 None 则使用 space。
        embedding_model: 若提供则用 SentenceTransformer 载入模型以得到 embedding_dim。
        reset_collection: 是否在写入前重置 collection（若支持）。

    Returns:
        dict: 简单汇总。
    """
    collection = collection_name or space

    # 加载 encoder 并获取维度
    encoder = None
    embedding_dim = None
    if embedding_model:
        encoder = SentenceTransformer(embedding_model)
        embedding_dim = encoder.get_sentence_embedding_dimension()
    else:
        # 如果不提供模型，交由 JointHandler.setup 接受显式 dim（暂不实现）
        raise ValueError("必须提供 embedding_model 来确定 embedding_dim，或修改函数以传入 embedding_dim。")

    joint = JointHandler(
        space_name=space,
        nebula_host=nebula_host,
        nebula_port=nebula_port,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection
    )
    joint.setup(embedding_dim=embedding_dim)

    processed_files = 0
    total_elements = 0
    try:
        if reset_collection:
            try:
                joint.milvus.reset_collection(embedding_dim=embedding_dim)
            except Exception as e:
                print(f"警告：重置 Milvus collection 失败：{e}")

        # 目录：递归查找以 _final_graph_with_vector.json 结尾的文件
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith('_final_graph_with_vector.json'):
                        json_path = os.path.join(root, file)
                        partition_name = os.path.splitext(file)[0].replace('_final_graph_with_vector', '')
                        with open(json_path, 'r', encoding='utf-8') as f:
                            final_graph = json.load(f)
                        print(f"--- 载入文件 {json_path}，共包含 {len(final_graph)} 个元素 ---")
                        joint.ingest_vector_data_bulk(final_graph, partition_name=partition_name)
                        processed_files += 1
                        total_elements += len(final_graph)

        # 单文件
        elif os.path.isfile(input_path):
            if input_path.endswith('_final_graph_with_vector.json'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    final_graph = json.load(f)
                partition_name = os.path.splitext(os.path.basename(input_path))[0].replace('_final_graph_with_vector', '')
                print(f"--- 载入文件 {input_path}，共包含 {len(final_graph)} 个元素 ---")
                joint.ingest_vector_data_bulk(final_graph, partition_name=partition_name)
                processed_files = 1
                total_elements = len(final_graph)
            else:
                msg = f"错误：文件 {input_path} 不是有效的 '_final_graph_with_vector.json' 文件。"
                print(msg)
                return {"success": False, "message": msg}
        else:
            msg = "错误：输入路径无效。请提供有效的文件夹路径或 JSON 文件路径。"
            print(msg)
            return {"success": False, "message": msg}

        # 建索引（若 JointHandler 提供）
        try:
            joint.milvus.create_vector_index()
        except Exception as e:
            print(f"警告：创建 Milvus 向量索引失败：{e}")

        return {
            "success": True,
            "processed_files": processed_files,
            "total_elements": total_elements,
            "space": space,
            "collection": collection
        }
    finally:
        try:
            joint.close()
        except Exception:
            pass
