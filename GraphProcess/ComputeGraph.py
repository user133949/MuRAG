import json
import networkx as nx
import csv
import sys
from pathlib import Path

def compute_graph(json_file: str, output_dir: str = "."):
    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()  # 有向图
    name_to_vid = {}
    vid_to_entity = {}

    # 先处理节点，建立 name → vid 映射，同时记录 vid → entityID
    for item in data:
        if "entityID" in item and "ref_doc_id" in item:
            vid = f"{item['ref_doc_id']}_{item['entityID']}"
            entity_id = str(item["entityID"])  # 转成字符串，避免列表作为 key
            G.add_node(vid, **item)
            vid_to_entity[vid] = entity_id

            # 建立 name 到 vid 的映射（只取第一个出现的）
            if "name" in item and item["name"] not in name_to_vid:
                name_to_vid[item["name"]] = vid

    # 再处理边，source/target 转成 vid
    for item in data:
        if "source" in item and "target" in item:
            src_name = item["source"]
            tgt_name = item["target"]
            src_vid = name_to_vid.get(src_name)
            tgt_vid = name_to_vid.get(tgt_name)

            if src_vid and tgt_vid:  # 只添加映射得到的有效边
                weight = item.get("relationship_strength", 1)
                G.add_edge(src_vid, tgt_vid, weight=weight, **item)

    # 计算 PageRank 和 Closeness
    pagerank_vid = nx.pagerank(G, weight="weight")
    closeness_vid = nx.closeness_centrality(G)

    # 转换为 entityID: 分数
    pagerank = {vid_to_entity[vid]: score for vid, score in pagerank_vid.items()}
    closeness = {vid_to_entity[vid]: score for vid, score in closeness_vid.items()}

    # 输出 CSV 文件
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pagerank_file = Path(output_dir) / "pagerank.csv"
    closeness_file = Path(output_dir) / "closeness.csv"

    with open(pagerank_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entityID", "pagerank"])
        for eid, score in pagerank.items():
            writer.writerow([eid, score])

    with open(closeness_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entityID", "closeness"])
        for eid, score in closeness.items():
            writer.writerow([eid, score])

    print(f"✅ PageRank 结果已保存到 {pagerank_file}")
    print(f"✅ Closeness 结果已保存到 {closeness_file}")

    # 返回两个 dict 构成的 list
    return pagerank, closeness


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ComputeGraph.py <json_file> [output_dir]")
    else:
        json_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        results = compute_graph(json_file, output_dir)
        print("📊 返回结果:", results)
