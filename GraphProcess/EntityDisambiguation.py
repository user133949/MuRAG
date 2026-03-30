import logging
import copy
import numpy as np
import re
import json
from typing import List, Dict
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

def _parse_chunk_ids(val):
    """
    接受 None / str / list，统一返回 list[list[int]] 形式。
    不抛异常，解析失败返回空列表。
    """
    if val is None:
        return []
    if isinstance(val, list):
        # 允许 list[list[int]] 或 list[tuple] 等
        out = []
        for item in val:
            try:
                out.append(list(item))
            except Exception:
                pass
        return out
    if isinstance(val, str):
        try:
            data = json.loads(val)
            return _parse_chunk_ids(data)
        except Exception:
            return []
    # 其他类型一律空
    return []

def _dump_chunk_ids(lst):
    """
    把 list[list[int]] 序列化成紧凑字符串，保证可回写到节点。
    """
    try:
        return json.dumps(lst, ensure_ascii=False)
    except Exception:
        return "[]"

def _normalize_name(name: str) -> str:
    """统一大小写与空白，便于鲁棒匹配"""
    if not isinstance(name, str):
        return ""
    # 去除多余的空白并转为小写
    s = re.sub(r"\s+", " ", name.strip())
    return s.lower()


def merge_graph_elements(graph: List[Dict], 
                         entity_id_map: Dict[str, str], 
                         name_map: Dict[str, str]) -> List[Dict]:
    """
    用 entity_id_map 决定节点合并；并把所有边的端点（name 字符串）重写成规范节点的 name。
    最后删除被合并节点、去重边，并过滤任何仍然指向不存在节点的边，避免悬挂边。
    """
    if not entity_id_map and not name_map:
        return graph

    # 1) 复制，避免副作用
    graph_copy = copy.deepcopy(graph)

    # 2) 先做一些索引：ID -> 节点对象 / name
    id2node = {}
    for e in graph_copy:
        if "relationship" not in e:
            eid = e.get("entityID")
            if isinstance(eid, str):
                id2node[eid] = e
    id2name = {eid: n.get("name") for eid, n in id2node.items() if isinstance(n.get("name"), str)}

    # 3) 基于 entity_id_map 推导出“被合并名 -> 规范名”的映射（name 级别）
    #    注意：这里才是把“ID 合并决策”落实到“边端点字符串重写”的关键
    name_map_from_ids = {}
    for merged_id, canonical_id in (entity_id_map or {}).items():
        old_name = id2name.get(merged_id)
        new_name = id2name.get(canonical_id)
        if old_name and new_name and old_name != new_name:
            name_map_from_ids[old_name] = new_name
    
     # ---------- 新增：在删除被合并节点之前，先把 chunkID 合并到规范节点 ----------
    if entity_id_map:
        # 先把每个规范ID准备好一个集合用于去重
        canon_to_chunks = defaultdict(set)

        # 1) 先把规范节点当前已有的 chunkID 解析进去
        for canon_id in set(entity_id_map.values()):
            canon_node = id2node.get(canon_id)
            if canon_node is None:
                continue
            canon_chunks = _parse_chunk_ids(canon_node.get("chunkID"))
            for ch in canon_chunks:
                try:
                    canon_to_chunks[canon_id].add(tuple(ch))
                except Exception:
                    pass

        # 2) 把每个被合并节点的 chunkID 汇入规范节点
        for merged_id, canonical_id in entity_id_map.items():
            merged_node = id2node.get(merged_id)
            if merged_node is None:
                continue
            merged_chunks = _parse_chunk_ids(merged_node.get("chunkID"))
            for ch in merged_chunks:
                try:
                    canon_to_chunks[canonical_id].add(tuple(ch))
                except Exception:
                    pass

        # 3) 把汇总后的 chunkID 写回规范节点（转回字符串）
        for canon_id, tup_set in canon_to_chunks.items():
            canon_node = id2node.get(canon_id)
            if canon_node is None:
                continue
            merged_list = [list(t) for t in sorted(tup_set)]
            canon_node["chunkID"] = _dump_chunk_ids(merged_list)
    # ---------- 新增结束 ------------------------------------------------------

    # 4) 合并外部传入的 name_map（如果有）与我们由 ID 推导出的 name_map
    combined_name_map = dict(name_map_from_ids)
    combined_name_map.update(name_map or {})  # 以外部传入的为准进行覆盖

    # 5) 再构造一个“规范化键”的 name_map，解决大小写/空白差异导致的未命中
    combined_name_map_norm = {_normalize_name(k): v 
                              for k, v in combined_name_map.items()
                              if isinstance(k, str) and isinstance(v, str)}

    # 6) 先重写边的端点（用原样命中，其次用规范化命中）
    def _rewrite_endpoint(ep: str) -> str:
        if not isinstance(ep, str):
            return ep
        # 6.1 原样命中
        if ep in combined_name_map:
            return combined_name_map[ep]
        # 6.2 规范化命中
        nep = _normalize_name(ep)
        if nep in combined_name_map_norm:
            return combined_name_map_norm[nep]
        return ep

    for element in graph_copy:
        if "relationship" in element:
            if "source" in element and "target" in element:
                source = element["source"]
                target = element["target"]
                # 过滤：如果 source 和 target 相同，表示自己指向自己，跳过这条边
                if source == target:
                    continue
                # 重写源和目标
                element["source"] = _rewrite_endpoint(source)
                element["target"] = _rewrite_endpoint(target)

    # 7) 删除被合并的节点（entity_id_map 的 key 是被删除者）
    merged_entity_ids = set(entity_id_map.keys()) if entity_id_map else set()
    survivors = []
    for elem in graph_copy:
        if "relationship" not in elem:
            # 节点
            if elem.get("entityID") in merged_entity_ids:
                continue  # 丢弃被合并节点
            survivors.append(elem)
        else:
            # 边暂不处理，统一放到后面过滤去重
            survivors.append(elem)

    # 8) 重新收集“存活节点”的名字集合，用于过滤孤立边
    node_names = {n.get("name") for n in survivors if "relationship" not in n and isinstance(n.get("name"), str)}

    # 9) 去重边（同一 (min(src,tgt), max(src,tgt), relationship) 保留 strength 最大），同时过滤端点不存在的边
    unique_best_edges = {}
    nodes_only, edges_only = [], []
    for e in survivors:
        if "relationship" not in e:
            nodes_only.append(e)
        else:
            src, tgt = e.get("source"), e.get("target")
            # 过滤：端点必须都在现存节点里
            if not (isinstance(src, str) and isinstance(tgt, str) and src in node_names and tgt in node_names):
                continue
            sig = tuple(sorted((src, tgt))) + (e.get("relationship"),)
            cur_strength = e.get("relationship_strength", 0)
            if sig not in unique_best_edges or cur_strength > unique_best_edges[sig].get("relationship_strength", 0):
                unique_best_edges[sig] = e

    cleaned_final_graph = nodes_only + list(unique_best_edges.values())
    return cleaned_final_graph

def disambiguate_by_exact_name(graph: List[Dict]) -> List[Dict]:
    """
    基于完全相同的实体名称进行合并。
    参数:
        graph (List[Dict]): 原始图谱。
    返回:
        List[Dict]: 消歧后的图谱。
    """
    STRUCTURAL_NODE_TYPES = {"SEGMENT ANCHOR NODE", "ORIGINAL_IMAGE", "EQUATION NODE", "TABLE NODE"}
    entity_groups = defaultdict(list)
    for element in graph:
        if "relationship" not in element and element.get("type") not in STRUCTURAL_NODE_TYPES:
            name = element.get("name", "").lower().strip()
            if name:
                entity_groups[name].append(element)

    entity_id_map = {}
    name_map = {} 

    for name, entities in entity_groups.items():
        if len(entities) > 1:
             # --- 关键修复：根据实体名称的长度，对列表进行升序排序 ---
            entities.sort(key=lambda e: len(e.get("name", "")))
            canonical_entity = entities[0]
            logging.info(f"发现重复实体组 '{name}', 将合并 {len(entities)} 个节点 -> 保留 entityID: {canonical_entity['entityID']}")
            for entity_to_merge in entities[1:]:
                entity_id_map[entity_to_merge["entityID"]] = canonical_entity["entityID"]
                name_map[entity_to_merge['name']] = canonical_entity['name']
    
    return merge_graph_elements(graph, entity_id_map, name_map)

def disambiguate_by_vector_similarity(graph: List[Dict], similarity_threshold=0.9) -> List[Dict]:
    """
    基于实体向量的余弦相似度进行合并。
    参数:
        graph (List[Dict]): 原始图谱。
        similarity_threshold (float): 相似度阈值。
    返回:
        List[Dict]: 消歧后的图谱。
    """
    STRUCTURAL_NODE_TYPES = {"SEGMENT ANCHOR NODE", "ORIGINAL_IMAGE", "EQUATION NODE", "TABLE NODE"}
    entities_to_cluster = []
    for element in graph:
        if "relationship" not in element and element.get("vector") and element.get("type") not in STRUCTURAL_NODE_TYPES:
            entities_to_cluster.append(element)

    if len(entities_to_cluster) < 2: return graph

    vectors = np.array([e["vector"] for e in entities_to_cluster])
    clustering = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=1 - similarity_threshold)
    cluster_labels = clustering.fit_predict(vectors)

    clusters = defaultdict(list)
    for i, entity in enumerate(entities_to_cluster):
        clusters[cluster_labels[i]].append(entity)
        
    entity_id_map = {}
    name_map = {}

    for cluster_id, entities in clusters.items():
        if len(entities) > 1:
            # --- 关键修复：根据实体名称的长度，对列表进行升序排序 ---
            entities.sort(key=lambda e: len(e.get("name", "")))
            canonical_entity = entities[0]
            entity_names = [e["name"] for e in entities]
            logging.info(f"发现语义相似簇, 将合并: {entity_names} -> 保留 '{canonical_entity['name']}' (entityID: {canonical_entity['entityID']})")
            for entity_to_merge in entities[1:]:
                entity_id_map[entity_to_merge["entityID"]] = canonical_entity["entityID"]
                name_map[entity_to_merge['name']] = canonical_entity['name']
    
    return merge_graph_elements(graph, entity_id_map, name_map)


def run_disambiguation_pipeline(graph: List[Dict], similarity_threshold=0.85) -> List[Dict]:
    """
    运行完整的实体消歧流水线。
    参数:
        graph (List[Dict]): 原始图谱。
        similarity_threshold (float): 相似度阈值。
    返回:
        List[Dict]: 消歧后的图谱。
    """
    if not graph: 
        logging.error("输入图谱为空，无法进行消歧")
        return []
    logging.info(f"原始图谱包含 {len(graph)} 个元素")

    graph_after_pass1 = disambiguate_by_exact_name(graph)
    print(f"\n第一轮消歧后，图谱剩余 {len(graph_after_pass1)} 个元素。")
    logging.info(f"第一轮消歧后，图谱剩余 {len(graph_after_pass1)} 个元素")

    final_graph = disambiguate_by_vector_similarity(graph_after_pass1, 
                                                    similarity_threshold=similarity_threshold)
    print(f"\n第二轮消歧后，图谱最终剩余 {len(final_graph)} 个元素。")
    logging.info(f"第二轮消歧后，图谱最终剩余 {len(final_graph)} 个元素")
    
    return final_graph