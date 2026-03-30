import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

def _select_top_k_chunks(scores: dict, top_k: int = 5) -> list[str]:
    """
    从 scores 字典中，按 value 排序，选取前 top_k 个 chunkID。

    参数:
        scores: {chunkID: score}
        top_k: 选取数量
    返回:
        [chunkID1, chunkID2, ...] 排好序的前k个结果
    """
    # 按value降序排序
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_id_list = [chunk_id for chunk_id, _ in sorted_items[:top_k]]
    return top_id_list

def _load_chunks(file_path: Path, 
                 pdf_name: str,
                 text_chunkID: list[str], 
                 special_chunkID: list[str]
                 )-> dict:
    """
    读取指定文件夹中最新的 OCR 解析文件，并根据统计字典提取对应的三元组
    部分节点的原文可能是多模态段落，这里会详细区分原始 chunk 是否是多模态节点

    参数:
    file_path: JSON 文件的文件夹路径
    text_chunkID: 文本 chunkID list [(page_idx, number), ...]
    special_chunkID: 多模态 chunkID list [(page_idx, number), ...]

    返回:
        chunks = {
            "text": {(page_idx, 序号): 对应json字典},
            "multimodal": {(page_idx, 序号): 对应json字典}
        }
    """
    json_files = list(file_path.glob("*.json"))
    if not json_files:
        print("未找到JSON文件")
        return defaultdict(list), ""
    # 取最 JSON 文件
    json_file_path = os.path.join(file_path, f"{pdf_name}_content_list.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 按页码分组
    pages = defaultdict(list)
    for item in data:
        pages[item['page_idx']].append(item)

    chunks = {"text": {}, "multimodal": {}}
    # 合并两个 chunkID 列表，标记来源
    for chunk_type, chunk_ids in [("text", text_chunkID), ("special", special_chunkID)]:
        for (page_idx, paragraph_idx) in chunk_ids:
            if page_idx in pages and paragraph_idx < len(pages[page_idx]):
                chunk = pages[page_idx][paragraph_idx]
                if chunk.get("type") == "text":
                    chunks["text"][(page_idx, paragraph_idx)] = chunk
                else:
                    chunks["multimodal"][(page_idx, paragraph_idx)] = chunk
    
    return chunks

def chunk_loader(ocr_json_path: Path, 
                 pdf_name: str,
                 scores_text: dict, 
                 scores_special: dict, 
                 top_k_text: int = 5, 
                 top_k_special: int = 4
                 ) -> dict:
    """"
    ocr_json_path: OCR 解析文件的文件夹路径
    scores_text: 文本来源的 chunk 得分字典 {chunkID: score}
    scores_special: 特殊来源(多模态/anchor) chunk 得分字典 {chunkID: score}
    top_k_text: 选取文本来源的 top_k 个chunk
    scores_special: 选取多模态来源的 top_k 个 chunk
    返回:
        chunks = {
            "text": {(page_idx, 序号): 对应json字典},
            "multimodal": {(page_idx, 序号): 对应json字典}
        }
    """
    # 按value降序排序
    top_text_chunkID = _select_top_k_chunks(scores_text, top_k_text)
    top_special_chunkID = _select_top_k_chunks(scores_special, top_k_special)

    # 根据段落type分别加载对应的段落内容
    chunks = _load_chunks(ocr_json_path, pdf_name, top_text_chunkID, top_special_chunkID)

    return chunks

def get_related_entities(v_id: dict, 
                         graph_data: List[Dict], 
                         hop: int = 1
                         ) -> Tuple[List[Dict], List[Dict]]:
    """
    根据检索到的 VID ，从知识图谱中获取与之相关的实体
    1. 只保留 entity 节点
    2. 只考虑 1-hop 和 2-hop 关系
    参数：
        entity_id (dict): 目标实体ID。
        graph_data (list): 包含实体和关系的知识图谱数据。
        hop (int): 跳数 默认为1。
    返回：
        list: 实体的列表。
    """
    # 构建实体ID到实体的映射
    combined = {"1-hop": [], "2-hop": []}
    for value in v_id.values():
        combined["1-hop"].extend(value["1-hop"])
        combined["2-hop"].extend(value["2-hop"])
    combined["2-hop"].extend(combined["1-hop"])

    # 只保留 entity ，并剔除掉 anchor节点
    entity_dict = {}
    for e in graph_data:
        if "relationship" in e:
            continue
        
        local_entity_id = e.get("entityID")
        ref_doc_id = e.get("ref_doc_id")

        if local_entity_id and ref_doc_id:
            # 构建与 neighbor_vids 中格式一致的组合VID
            vid_key = f"{ref_doc_id}_{local_entity_id}"
            entity_dict[vid_key] = e

    # 根据跳数获取相关实体
    hop_key = f"{hop}-hop"
    target_vids = combined.get(hop_key, [])
    anchor_nodes = []
    other_nodes = []
    for target_vid in target_vids:
        # 检查目标VID是否存在于我们的字典中
        if target_vid in entity_dict:
            # 获取完整的实体对象
            entity = entity_dict[target_vid]
            
            # 根据类型进行分类
            if entity.get("type") == "SEGMENT ANCHOR NODE":
                anchor_nodes.append(entity)
            else:
                other_nodes.append(entity)

    return anchor_nodes, other_nodes



def select_round_robin_top_by_pages(
    scores_anchor: Dict[Tuple[int,int], float],
    scores_text: Dict[Tuple[int,int], float],
    scores_mm: Dict[Tuple[int,int], float],
    max_pages: int = 10
) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[int]]:
    """
    三条榜单各自降序，然后轮流取（anchor -> text -> mm -> anchor ...），
    直到已选 chunk 来自 max_pages 个不同页码，或三条榜单都耗尽。

    返回：
        selected_anchor, selected_text, selected_mm, selected_pages_list
    其中每个 selected_* 是 chunkID 列表 (page_idx, paragraph_idx)；
    selected_pages_list 是已选到的页码（最多 max_pages），以列表形式返回。
    """

    def sorted_keys(scores: Dict[Tuple[int,int], float]) -> List[Tuple[int,int]]:
        return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    lists = [
        sorted_keys(scores_anchor),
        sorted_keys(scores_text),
        sorted_keys(scores_mm)
    ]
    names = ['anchor', 'text', 'mm']
    idxs = [0, 0, 0]
    selected = {'anchor': [], 'text': [], 'mm': []}
    pages_seen: set[int] = set()
    pages_order: list[int] = []
    exhausted = [False, False, False]

    # 如果所有榜单都为空
    if not any(lists):
        return [], [], [], []

    while len(pages_seen) < max_pages and not all(exhausted):
        progressed = False
        for i in range(3):
            if exhausted[i]:
                continue
            lst = lists[i]
            j = idxs[i]
            # 跳过已经被选过的 chunk
            while j < len(lst) and lst[j] in (selected['anchor'] + selected['text'] + selected['mm']):
                j += 1
            if j < len(lst):
                cid = lst[j]
                selected[names[i]].append(cid)
                idxs[i] = j + 1
                progressed = True
                # 记录页码
                if cid[0] not in pages_seen:
                    pages_seen.add(cid[0])
                    pages_order.append(cid[0])
                # 达到页数上限则退出
                if len(pages_seen) >= max_pages:
                    break
            else:
                exhausted[i] = True
        if not progressed:
            break

    return selected['anchor'], selected['text'], selected['mm'], pages_order[:max_pages]

def parse_raw_evidence_to_gold_zero(qa: Dict[str, Any]) -> List[int]:
    """
    将 qa 中的 evidence_page / evidence_pages 规范化为 0-based 的页码列表 gold_zero。
    使用你提供的规范化逻辑（尽量兼容字符串、列表、数字等）。
    """
    raw_evidence = None
    if "evidence_page" in qa:
        raw_evidence = qa.get("evidence_page")
    elif "evidence_pages" in qa:
        raw_evidence = qa.get("evidence_pages")

    gold_zero: List[int] = []
    if raw_evidence:
        if isinstance(raw_evidence, str):
            try:
                parsed = json.loads(raw_evidence.replace("'", '"'))
                raw_list = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                cleaned = raw_evidence.strip("[] ").replace("'", "").replace('"', "")
                raw_list = [x.strip() for x in cleaned.split(",") if x.strip()]
        elif isinstance(raw_evidence, (list, tuple)):
            raw_list = list(raw_evidence)
        elif isinstance(raw_evidence, (int, float)):
            raw_list = [int(raw_evidence)]
        else:
            raw_list = []

        for v in raw_list:
            try:
                vi = int(v)
                gold_zero.append(max(0, vi - 1))
            except Exception:
                s = str(v)
                digits = ''.join(ch for ch in s if ch.isdigit())
                if digits:
                    vi = int(digits)
                    gold_zero.append(max(0, vi - 1))
    return gold_zero

def extract_pages_from_chunkid(chunkid: Any) -> List[int]:
    """
    从 chunkID 字段提取出每一对的第一个数字（页码列表）。
    chunkid 可能的形式：
      - 真正的 list/tuple: [[0,3],[14,5]]
      - 字符串: "[[0, 3], [14, 5]]" / "'[[0,3],[14,5]]'"
      - 其它奇怪的字符串形式，使用正则提取 [a,b] 的 a
    返回值为 int 列表（不做 -1 或 +1 的改变，假设 chunkID 中数字为 0-based）。
    """
    pages: List[int] = []
    # 如果已经是 list/tuple
    if isinstance(chunkid, (list, tuple)):
        for item in chunkid:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                try:
                    pages.append(int(item[0]))
                except Exception:
                    # 忽略无法转换的项
                    continue
            else:
                # 如果是简单数字或字符串，尝试直接解析
                try:
                    pages.append(int(item))
                except Exception:
                    continue
        return pages

    # 如果是字符串，先尝试 json.loads
    if isinstance(chunkid, str):
        s = chunkid.strip()
        # 先尝试把单引号替换并 json 解析
        try:
            parsed = json.loads(s.replace("'", '"'))
            return extract_pages_from_chunkid(parsed)
        except Exception:
            # 使用正则提取所有形如 [num, ...] 或 (num, ...) 的首个数字
            # 这会匹配 [0, 3]、[14,5]、(2,1) 等
            pair_patterns = re.findall(r'\[\s*([-]?\d+)\s*,', s)
            if not pair_patterns:
                # 也尝试匹配单个数字（退化情况）
                single_nums = re.findall(r'([-]?\d+)', s)
                for n in single_nums:
                    try:
                        pages.append(int(n))
                    except Exception:
                        continue
            else:
                for n in pair_patterns:
                    try:
                        pages.append(int(n))
                    except Exception:
                        continue
            return pages

    # 其余类型尝试强转为 int
    try:
        pages.append(int(chunkid))
    except Exception:
        pass
    return pages

def load_graph_and_filter(dataset_name: str,
                          pdf_name: str,
                          qa: Dict[str, Any],
                          graph_root: str = "/home/hdd/MRAG/Dataset/GraphCache") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    主函数：
      - dataset_name, pdf_name: 用于定位 JSON 文件
      - qa: 包含 evidence_page / evidence_pages 的字典（会调用 parse_raw_evidence_to_gold_zero）
      - 返回 (anchor_nodes, entity_nodes)
    """
    path = Path(graph_root) / dataset_name / "FinalGraph" / f"{pdf_name}_final_graph.json"
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            # 有时文件是一行多个 json 对象或格式稍怪，可以尝试逐行解析
            f.seek(0)
            lines = f.readlines()
            combined = ''.join(lines)
            try:
                data = json.loads(combined)
            except Exception:
                raise RuntimeError(f"Failed to parse JSON file {path}: {e}")

    # 生成 gold_zero（0-based 页码）
    gold_zero = parse_raw_evidence_to_gold_zero(qa)

    # ⚠️ 修改部分：如果 evidence_page 为空，则默认取前5页
    if not gold_zero:
        gold_zero = list(range(5))  # [0,1,2,3,4]

    anchor_nodes: List[Dict[str, Any]] = []
    entity_nodes: List[Dict[str, Any]] = []

    # data 可能是 list（节点/边数组）或 dict（包含 "nodes" 字段）
    items: List[Dict[str, Any]]
    if isinstance(data, dict) and ("nodes" in data or "graph" in data):
        if "nodes" in data and isinstance(data["nodes"], list):
            items = data["nodes"]
        elif "graph" in data and isinstance(data["graph"], list):
            items = data["graph"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for entry in items:
        if not isinstance(entry, dict):
            continue
        # 只处理包含 chunkID 的第一类形式
        if "chunkID" not in entry:
            continue

        chunkid = entry.get("chunkID")
        pages = extract_pages_from_chunkid(chunkid)
        if not pages:
            continue

        # 如果任何 page 在 gold_zero 中，则认为该节点相关
        matched = any((p in gold_zero) for p in pages)
        if not matched:
            continue

        typ = entry.get("type", "")
        if isinstance(typ, str) and typ.strip().upper() == "SEGMENT ANCHOR NODE":
            anchor_nodes.append(entry)
        else:
            entity_nodes.append(entry)

    return anchor_nodes, entity_nodes
