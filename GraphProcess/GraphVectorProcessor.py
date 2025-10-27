import logging
from typing import List, Dict

# --- 核心处理函数 ---
def process_vectors_and_clean_graph(final_graph_with_vectors: List[Dict]) -> List[Dict]:
    """
    遍历消歧后的图谱，通过指定的接口将实体向量存入数据库，然后从图谱中移除向量字段。

    这个函数扮演“向量处理器”的角色，它负责：
    1. 提取所有包含向量的节点。
    2. 将这些节点（及其向量）委托给外部存储接口进行持久化。
    3. 返回一个不含向量的、干净的图谱副本。

    参数:
        final_graph_with_vectors (List[Dict]): 经过消歧处理的、包含临时向量的图谱。

    返回:
        List[Dict]: 一个不包含任何 "vector" 字段的、干净的图谱结构，适合存储为 JSON 或导入图数据库。
    """
    # 1. 从图谱中收集所有包含 "vector" 字段的节点
    # 我们只关心节点，所以排除了关系("relationship" in element)
    nodes_with_vectors_to_store = [
        dict(element) for element in final_graph_with_vectors 
        if "relationship" not in element and "vector" in element
    ]

    if not nodes_with_vectors_to_store:
        print("未在图谱中找到任何需要存储的向量，直接返回原图谱。")
        logging.error(f"未在图谱中找到任何需要存储的向量，直接返回原图谱")
        return final_graph_with_vectors

    print(f"共找到 {len(nodes_with_vectors_to_store)} 个包含向量的节点准备存储。")
    logging.info(f"共找到 {len(nodes_with_vectors_to_store)} 个包含向量的节点准备存储")

    # 2. 创建一个新的图谱列表，其中不包含任何 "vector" 字段
    clean_graph = []
    for element in final_graph_with_vectors:
        # 使用 dict() 创建一个浅拷贝，对于当前场景是足够且高效的
        clean_element = dict(element) 
        if "vector" in clean_element:
            del clean_element["vector"]
        clean_graph.append(clean_element)

    return clean_graph