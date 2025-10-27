import ast
import csv
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

def _graph_entities_loader(graph_data: list) -> list:
    """
    载入 JSON 格式的 graph 并筛选出实体部分
    返回:
        实体列表
    """
    entities = []
    for item in graph_data:
        # 判断是不是实体
        if all(key in item for key in ["name", "type", "description", "entityID"]):
            entities.append(item)
    
    return entities

def _compute_entity_similarities(entities: list, 
                                 query: str, 
                                 encoder: SentenceTransformer, 
                                 batch_size: int = 2   # ✅ 默认批大小，可调小 最大占用显存为(batch+1)*4096
                                 ) -> dict:
    """
    计算 query 和子图中各个 entity 的相似度（批处理，避免显存爆炸）。
    """
    if not entities:   # ✅ 如果实体为空，直接返回空字典
        return {}

    entity_texts = [
        f"{e['name']} {e['type']} {e['description']}" for e in entities if "entityID" in e
    ]
    entity_ids = [e["entityID"] for e in entities if "entityID" in e]

    if not entity_texts:   # ✅ 如果没有有效文本，返回空字典
        return {}

    # 计算 query embedding
    query_embedding = encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    similarities = {}
    # ✅ 分批计算，避免显存爆炸
    for i in range(0, len(entity_texts), batch_size):
        batch_texts = entity_texts[i:i+batch_size]
        batch_ids = entity_ids[i:i+batch_size]

        batch_embeddings = encoder.encode(
            batch_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False
        )

        # 计算相似度
        batch_scores = util.cos_sim(query_embedding, batch_embeddings)[0]

        # 保存结果
        for j, eid in enumerate(batch_ids):
            similarities[eid] = float(batch_scores[j])

        # ✅ 主动清理显存，避免累计
        del batch_embeddings, batch_scores
        torch.cuda.empty_cache()

    return similarities

def _chunk_statistics(entities:list) -> tuple[dict, dict, dict, dict]:
    """
    从JSON格式的实体合集中提取实体的 entityID 前两位 (chunkID)
    统计唯一值及出现次数

    返回:
        text_entities: {chunkID: [entityID列表]} 文本chunk具体实体
        special_entities: {chunkID: [entityID列表]} 多模态chunk具体实体
    """
    text_chunk = []
    special_chunk = []
    text_entities = defaultdict(list)
    special_entities = defaultdict(list)

    for item in entities:
        if "entityID" in item:
            entity_id = ast.literal_eval(item["entityID"])
            chunk_id_list = ast.literal_eval(item["chunkID"])

            if isinstance(entity_id[2], (int, float)):  # 文本chunk
                text_chunk.append(chunk_id_list)
                for cid in chunk_id_list:
                    text_entities[tuple(cid)].append(str(entity_id)) # 转成str
            elif (isinstance(entity_id[2], str)):  # 特殊chunk（多模态或anchor）
                special_chunk.append(chunk_id_list)
                for cid in chunk_id_list:
                    special_entities[tuple(cid)].append(str(entity_id))
    # 统计并转成普通字典
    return dict(text_entities), dict(special_entities)

def _load_pagerank_dict(csv_file_path: str) -> dict:
    """
    读取CSV文件 将 _id 和 _pagerank 组成字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        dict: {_id(str): _pagerank(float)}
    """
    pagerank_dict = {}
    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pagerank_dict[row["_id"]] = float(row["_pagerank"])
    except FileNotFoundError:
        print(f"未找到CSV文件: {csv_file_path}")
    except Exception as e:
        print(f"读取CSV文件出错: {e}")

    return pagerank_dict

def _load_closeness_dict(csv_file_path: str) -> dict:
    """
    读取CSV文件 将 _id 和 _closeness 组成字典。
    
    参数:
        csv_file_path (str): CSV文件路径
    
    返回:
        dict: {_id(str): _closeness(float)}
    """
    closeness_dict = {}
    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                closeness_dict[row["_id"]] = float(row["_closeness"])
    except FileNotFoundError:
        print(f"未找到CSV文件: {csv_file_path}")
    except Exception as e:
        print(f"读取CSV文件出错: {e}")

    return closeness_dict

def _relevance_score(similarities: dict, 
                     chunk_entities: dict, 
                     alpha: float
                     ) -> dict:
    """
    计算 chunk relevance score。
    """
    if not chunk_entities:   # ✅ 没有 chunk 直接返回空
        return {}

    sim_scores = defaultdict(float)
    num_socres = defaultdict(float)
    for chunk_id, entity_list in chunk_entities.items():
        for eid in entity_list:
            sim = similarities.get(eid, 0.0)
            sim_scores[chunk_id] += sim
            num_socres[chunk_id] += 1

    if not sim_scores:   # ✅ 避免除零
        return {}

    total1 = sum(sim_scores.values())
    similarities_prob = {node: (val / total1 if total1 > 0 else 0.0) for node, val in sim_scores.items()}
    total2 = sum(num_socres.values())
    chunk_entities_prob = {node: (val / total2 if total2 > 0 else 0.0) for node, val in num_socres.items()}

    return {node: alpha * similarities_prob.get(node, 0.0) + (1 - alpha) * chunk_entities_prob.get(node, 0.0)
            for node in sim_scores.keys()}

def _structure_score(pagerank_dict: dict, 
                     closeness_dict: dict, 
                     chunk_entities: dict, 
                     beta: float
                     ) -> dict:
    """
    根据每个 chunk 对应节点的 PageRank 值和 Closeness Centrality 值，
    根据权重，计算每个 chunk 的结构得分 Score(c)。
    参数:
        pagerank_dict: {entityID(str): pagerank(float)}
        closeness_dict: {entityID(str): closeness(float)}
        chunk_entities: {chunkID(tuple): [entityID(str),...]} 每个chunk包含的entity
        beta: PageRank 和 Closeness Centrality 奖励的权重, 取值范围 [0, 1]
    """
    
    # closeness 值归一化为和为1的概率值，和 pagerank 归一到同一量级
    total = sum(closeness_dict.values())
    closeness_prob_dict = {node: val/total for node, val in closeness_dict.items()}

    chunk_scores = {}
    for chunk_id, entity_list in chunk_entities.items():
        total = 0.0
        for eid in entity_list:
            pagerank = pagerank_dict.get(eid, 0.0)   # 没有则视为0
            closeness = closeness_prob_dict.get(eid, 0.0)    # 没有则视为0
            total += beta * pagerank + (1-beta) * closeness
        chunk_scores[chunk_id] = total

    return chunk_scores

def _rank_scores(dict1, dict2, lambda_value):
    """
    对两个字典进行排名，返回一个新的字典，其中键是原始字典的键，值是排名。
    
    参数：
    - dict1: 第一个字典
    - dict2: 第二个字典
    
    返回：
    - final_rank_dict: key 对应的综合排名字典
    """
    # 对 dict 排名，值越大排名值越大
    sorted_items1 = sorted(dict1.items(), key=lambda x: x[1])
    rank1 = {k: i+1 for i, (k, _) in enumerate(sorted_items1)}
    
    sorted_items2 = sorted(dict2.items(), key=lambda x: x[1])
    rank2 = {k: i+1 for i, (k, _) in enumerate(sorted_items2)}
    
    # 对每个 key 的排名相加
    final_rank_dict = {k: lambda_value * rank1[k] + 
                       (1-lambda_value) * rank2[k] for k in dict1.keys()}
    
    return final_rank_dict

def chunk_score(query: str, 
                global_data: list, 
                local_data: list, 
                pagerank: dict | str,   
                closeness: dict | str, 
                encoder_model: SentenceTransformer, 
                alpha=0.5, 
                beta=0.5,  
                lam=0.5
                ) -> tuple[dict, dict, dict]:
    """
    对 text 和 multimodal 段落计算 chunk relevance scores
    """
    # 1. 提取实体
    anchor_entities = _graph_entities_loader(global_data) if global_data else []
    local_entities = _graph_entities_loader(local_data) if local_data else []

    # 2. 统计 chunk 实体
    _, sum_chunk_entities = _chunk_statistics(anchor_entities) if anchor_entities else ({}, {})
    text_chunk_entities, mm_chunk_entities = _chunk_statistics(local_entities) if local_entities else ({}, {})

    # 3. 计算 query 相似度
    anchor_sim = _compute_entity_similarities(anchor_entities, query, encoder_model)
    local_sim = _compute_entity_similarities(local_entities, query, encoder_model)

    # 4. 处理 pagerank / closeness
    pagerank_dict = pagerank if isinstance(pagerank, dict) else (_load_pagerank_dict(pagerank) if isinstance(pagerank, str) else {})
    closeness_dict = closeness if isinstance(closeness, dict) else (_load_closeness_dict(closeness) if isinstance(closeness, str) else {})

    # 5. relevance scores
    relevance_scores_anchor = _relevance_score(anchor_sim, sum_chunk_entities, alpha)
    relevance_scores_text = _relevance_score(local_sim, text_chunk_entities, alpha)
    relevance_scores_mm = _relevance_score(local_sim, mm_chunk_entities, alpha)

    # 6. structure scores
    structure_scores_anchor = _structure_score(pagerank_dict, closeness_dict, sum_chunk_entities, beta) if sum_chunk_entities else {}
    structure_scores_text = _structure_score(pagerank_dict, closeness_dict, text_chunk_entities, beta) if text_chunk_entities else {}
    structure_scores_mm = _structure_score(pagerank_dict, closeness_dict, mm_chunk_entities, beta) if mm_chunk_entities else {}

    # 7. 综合排序
    scores_anchor = _rank_scores(relevance_scores_anchor, structure_scores_anchor, lam) if relevance_scores_anchor else {}
    scores_text = _rank_scores(relevance_scores_text, structure_scores_text, lam) if relevance_scores_text else {}
    scores_mm = _rank_scores(relevance_scores_mm, structure_scores_mm, lam) if relevance_scores_mm else {}

    return scores_anchor, scores_text, scores_mm

def ranks_to_chunk_scores(anchor_nodes: list, entity_nodes: list):
    """
    用已有的降序列表（anchor_nodes, entity_nodes）生成三类 chunk score dict：
    返回 (scores_anchor, scores_text, scores_mm)，每个为 { chunk_tuple: score }。
    规则：
      - 列表中越靠前的实体分数越高，分数 = (N - idx) / N
      - 每个实体可能属于多个 chunk（chunkID 是 list），把实体分数分配到每个 chunk
      - chunk 的最终分数取其包含实体分数的 max（保留最优实体的影响）
      - type 判定与 _chunk_statistics 保持一致：entityID 第三位为数字 -> text，否则为 mm（字符串）
    """
    def _assign_scores_from_list(nodes):
        """返回 dict: chunk_tuple -> list of entity scores"""
        chunk_scores_lists = defaultdict(list)
        if not nodes:
            return chunk_scores_lists

        N = len(nodes)
        for idx, item in enumerate(nodes):
            # 位置分数（降序列表：第一个得1.0，最后一个得1/N）
            score = (N - idx) / N

            # 抽取 chunkID
            try:
                chunk_id_list = ast.literal_eval(item.get("chunkID", "[]"))
            except Exception:
                # 不能解析则跳过该实体
                continue

            # chunk_id_list 预期为 list of [a,b] 之类
            for cid in chunk_id_list:
                try:
                    key = tuple(cid)
                except Exception:
                    # 如果 cid 不是可转为 tuple 的，跳过
                    continue
                chunk_scores_lists[key].append(score)

        return chunk_scores_lists

    # 1) anchor: 直接把 anchor_nodes 的排名映射到 chunk
    anchor_chunk_score_lists = _assign_scores_from_list(anchor_nodes)

    # 2) entity_nodes 需要按类型拆分 text / mm
    text_chunk_score_lists = defaultdict(list)
    mm_chunk_score_lists = defaultdict(list)

    if entity_nodes:
        N_local = len(entity_nodes)
        for idx, item in enumerate(entity_nodes):
            score = (N_local - idx) / N_local
            # 解析 entityID 查第三位
            is_text = None
            try:
                eid = ast.literal_eval(item.get("entityID", "[]"))
                if isinstance(eid, (list, tuple)) and len(eid) > 2:
                    is_text = isinstance(eid[2], (int, float))
            except Exception:
                # 如果解析失败，默认当 text 处理（或你可以改为 None 跳过）
                is_text = True

            try:
                chunk_id_list = ast.literal_eval(item.get("chunkID", "[]"))
            except Exception:
                continue

            for cid in chunk_id_list:
                try:
                    key = tuple(cid)
                except Exception:
                    continue
                if is_text:
                    text_chunk_score_lists[key].append(score)
                else:
                    mm_chunk_score_lists[key].append(score)

    # 转为最终 chunk -> 单一 score（取 max）
    def _finalize(lists_dict):
        return {k: max(v) for k, v in lists_dict.items()} if lists_dict else {}

    scores_anchor = _finalize(anchor_chunk_score_lists)
    scores_text = _finalize(text_chunk_score_lists)
    scores_mm = _finalize(mm_chunk_score_lists)

    return scores_anchor, scores_text, scores_mm