import json
from typing import List, Dict, Optional, Tuple, Set
import numpy as np


def _load_graph(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_graph(graph: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def _infer_ref_doc_id(graph: List[Dict]) -> str:
    for it in graph:
        if isinstance(it, dict) and isinstance(it.get("ref_doc_id"), str) and it["ref_doc_id"].strip():
            return it["ref_doc_id"]
    return ""


def _existing_edge_keys(graph: List[Dict]) -> Set[Tuple[str, str, str]]:
    keys = set()
    for it in graph:
        if isinstance(it, dict) and "source" in it and "target" in it and "relationship" in it:
            keys.add((it["source"], it["target"], it["relationship"]))
    return keys


def _create_similarity_anchor_edges(
    pdf_name: str,
    a: str,
    b: str,
    strength: int,
    relationship: str,
) -> List[Dict]:
    return [
        {
            "source": a,
            "target": b,
            "relationship": relationship,
            "relationship_strength": strength,
            "ref_doc_id": pdf_name,
        },
        {
            "source": b,
            "target": a,
            "relationship": relationship,
            "relationship_strength": strength,
            "ref_doc_id": pdf_name,
        },
    ]


def add_similarity_edges_for_anchor_nodes(
    graph_with_vector_path: str,
    graph_no_vector_path: str,
    output_with_vector_path: str,
    output_no_vector_path: str,
    similarity_threshold: float = 0.85,
    strength: int = 3,
    relationship: str = "edges between similar segment anchor nodes",
    block_size: int = 512,
) -> int:
    """
    1. 读取 with_vector graph，用向量计算相似度
    2. 生成新边（双向）
    3. 同步添加到 with_vector / no_vector 两个 graph
    4. 分别写入 final 文件
    """

    graph_vec = _load_graph(graph_with_vector_path)
    graph_no_vec = _load_graph(graph_no_vector_path)

    pdf_name = _infer_ref_doc_id(graph_vec)

    # ---- 取 anchor 节点 ----
    anchors = [
        it for it in graph_vec
        if it.get("type") == "SEGMENT ANCHOR NODE"
        and isinstance(it.get("name"), str)
        and isinstance(it.get("vector"), list)
        and len(it["vector"]) > 0
    ]

    if len(anchors) < 2:
        _save_graph(graph_vec, output_with_vector_path)
        _save_graph(graph_no_vec, output_no_vector_path)
        return 0

    names = [a["name"] for a in anchors]

    vecs = np.array([a["vector"] for a in anchors], dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    existing_keys = _existing_edge_keys(graph_vec)

    new_edges: List[Dict] = []
    n = len(names)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        A = vecs[i0:i1]

        for j0 in range(i0, n, block_size):
            j1 = min(j0 + block_size, n)
            B = vecs[j0:j1]

            sim = A @ B.T

            for ii in range(i1 - i0):
                gi = i0 + ii
                jj_start = max(0, gi + 1 - j0)

                for jj in range(jj_start, j1 - j0):
                    if sim[ii, jj] < similarity_threshold:
                        continue

                    gj = j0 + jj
                    a, b = names[gi], names[gj]

                    k1 = (a, b, relationship)
                    k2 = (b, a, relationship)

                    if k1 in existing_keys or k2 in existing_keys:
                        continue

                    edges = _create_similarity_anchor_edges(
                        pdf_name, a, b, strength, relationship
                    )

                    new_edges.extend(edges)
                    existing_keys.add(k1)
                    existing_keys.add(k2)

    # ---- 同步写入两个 graph ----
    graph_vec.extend(new_edges)
    graph_no_vec.extend(new_edges)

    _save_graph(graph_vec, output_with_vector_path)
    _save_graph(graph_no_vec, output_no_vector_path)

    return len(new_edges)


if __name__ == "__main__":
    n = add_similarity_edges_for_anchor_nodes(
        graph_with_vector_path="xxx_graph_with_vector.json",
        graph_no_vector_path="xxx_graph.json",
        output_with_vector_path="xxx_final_graph_with_vector.json",
        output_no_vector_path="xxx_final_graph.json",
        similarity_threshold=0.8
    )
    print("新增边数量:", n)
