import os
import sys
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
try:
    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT))
except NameError:
    ROOT = Path(".").resolve()

from GraphProcess.GraphEstablish import GraphProcessor
from GraphProcess.AnchorSimilarityEdge import add_similarity_edges_for_anchor_nodes
from LLM.LLMclient import ChatModel
from Database.joint import JointHandler
from MinerU.DocumentParser import parse_doc
from Retrieval.ChunkRetriever import get_related_entities, _load_chunks
from Retrieval.RelevanceScore import chunk_score
from Retrieval.SystemParameter import get_system_parameter, extract_weights
from Generation.Generator import unimodal_generator, multimodal_generator
from Generation.LLMJudge import judge_answer, score_answer
from GraphProcess.ComputeGraph import compute_graph
from Database.nebula_insert import ingest_graph_data
from Database.milvus_insert import ingest_vector_data


def display_result(document_name: str, question: str, expected: str, response: str, judgment: str, score: float, modal: str):
    """在控制台以清晰格式展示结果"""
    sep = "=" * 100
    print(sep)
    print(f"文档名: {document_name}")
    print(f"问题: {question}")
    print(f"期望答案: {expected}")
    print(f"模型类型: {modal}")
    print("-" * 100)
    print("模型回答:")
    print(response if response else "(空)")
    print("-" * 100)
    print(f"判断（Judge 返回）: {judgment}")
    print(f"评分: {score:.2f}")
    print(sep)


def main():
    SPACE = "MuRAG"
    model = "qwen-plus"
    reasoning_model = False
    vl_model = "qwen-vl-plus"
    api_key = "sk-xxx"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 文档与 QA（示例）
    pdf_name = "06f4135ecdf57596ac473ab26bbe4b21"
    question = "What is the increase/ (decrease) in Net income (loss) from 2018 to 2019?"
    expected_answer = "30.5"

    base_dir = Path("data")
    pdf_file = base_dir / "pdf" / f"{pdf_name}.pdf"
    parsed_root = base_dir / "parsed"
    parsed_pdf_dir = parsed_root / pdf_name

    graph_original_dir = Path("data") / "Graph" / "OriginalGraph"
    graph_final_dir = Path("data") / "Graph" / "FinalGraph"

    output_root = Path("Generation") / "Output"
    output_root.mkdir(parents=True, exist_ok=True)

    judge_dir = Path("Generation") / "Judge_result"
    judge_dir.mkdir(parents=True, exist_ok=True)
    judge_file = judge_dir / f"{pdf_name}_judge.csv"

    ocr_cache_root = parsed_root

    rag_file = output_root / f"{pdf_name}_rag.json"

    content_dir = parsed_pdf_dir / "auto"

    # ----- 初始化模型与数据库连接 -----
    encoder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    chatLLM = ChatModel(model=model, reasoning_model=reasoning_model, api_key=api_key, base_url=base_url, temperature=1.0)
    chatVLM = ChatModel(model=vl_model, reasoning_model=False, api_key=api_key, base_url=base_url, temperature=1.0)

    joint = JointHandler(space_name=SPACE,
                         nebula_host="127.0.0.1", nebula_port=9669,
                         milvus_host="127.0.0.1", milvus_port=19530,
                         collection_name=SPACE)
    # joint.setup 需要 embedding_dim
    joint.setup(embedding_dim=encoder.get_sentence_embedding_dimension())

    # ----- 解析文档（如果尚未解析） -----
    if parsed_pdf_dir.exists():
        print(f"PDF 已解析：{parsed_pdf_dir}")
    else:
        print(f"开始解析 PDF: {pdf_file}")
        parse_doc({pdf_file}, str(ocr_cache_root), backend="pipeline")
        if not parsed_pdf_dir.exists():
            print("解析失败，退出")
            return

    # ----- 构建图并入库 -----
    print("开始构建图。")
    processor = GraphProcessor(api_key=api_key)
    processor.contentlist_to_graph(pdf_name=pdf_name)
    add_similarity_edges_for_anchor_nodes(
        graph_with_vector_path = graph_final_dir / f"{pdf_name}_graph_with_vector.json",
        graph_no_vector_path = graph_final_dir / f"{pdf_name}_graph.json",
        output_with_vector_path = graph_final_dir / f"{pdf_name}_final_graph_with_vector.json",
        output_no_vector_path = graph_final_dir / f"{pdf_name}_final_graph.json",
        similarity_threshold = 0.8
    )
    json_path_novector = graph_final_dir / f"{pdf_name}_final_graph.json"
    json_path_withvector = graph_final_dir / f"{pdf_name}_final_graph_with_vector.json"

    print("开始将图数据写入数据库（Nebula / Milvus）...")
    ingest_graph_data(
        space=SPACE,
        input_path=str(json_path_novector),
        nebula_host="127.0.0.1",
        nebula_port=9669,
        embedding_dim=1024,
        entity_batch=400,
        relation_batch=400,
        reset=True,
    )

    ingest_vector_data(
        space=SPACE,
        input_path=str(json_path_withvector),
        nebula_host="127.0.0.1",
        nebula_port=9669,
        milvus_host="127.0.0.1",
        milvus_port=19530,
        collection_name=SPACE,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        reset_collection=True,
    )

    print(f"处理文件: {pdf_name}")
    if not joint.milvus.load_partition(pdf_name):
        print(f"无法加载 Milvus 分区 {pdf_name}，请检查 Milvus 状态。")
        return

    try:
        # 计算 graph 指标（pagerank, closeness）
        pagerank, closeness = compute_graph(str(json_path_novector), str(output_root))

        if not (json_path_novector.exists() and json_path_withvector.exists()):
            print("图文件缺失，退出")
            return

        with open(json_path_novector, 'r', encoding='utf-8') as f:
            final_graph = json.load(f)
        with open(json_path_withvector, 'r', encoding='utf-8') as f:
            final_graph_with_vectors = json.load(f)

        query = question
        q_vector = encoder.encode(query, convert_to_tensor=False)

        # 向 Milvus 搜索邻居
        search_results = joint.search_neighbors_by_vector(q_vector.tolist(), top_k=80, top_m=20, top_n=30, partition=pdf_name)

        anchor_nodes, entity_nodes = get_related_entities(v_id=search_results, graph_data=final_graph)

        # 获取 RAG 参数并提取权重
        rag_parameters = get_system_parameter(model=model, reasoning_model=reasoning_model, query=query, api_key=api_key, base_url=base_url)
        parameters = extract_weights(rag_parameters)
        lam = parameters.get("lam")

        # 计算 chunk 的 score
        scores_anchor, scores_text, scores_mm = chunk_score(query=query, global_data=anchor_nodes, local_data=entity_nodes,
                                                            pagerank=pagerank, closeness=closeness,
                                                            encoder_model=encoder, lam=lam)

        def _sorted_keys_by_score(scores: dict):
            return [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

        anchor_sorted = _sorted_keys_by_score(scores_anchor)
        mm_sorted = _sorted_keys_by_score(scores_mm)
        text_sorted = _sorted_keys_by_score(scores_text)

        selected = []
        selected_anchor_ids = []
        selected_mm_ids = []
        selected_text_ids = []

        for cid in anchor_sorted:
            if len(selected_anchor_ids) >= 8:
                break
            if cid not in selected:
                selected_anchor_ids.append(cid)
                selected.append(cid)

        for cid in mm_sorted:
            if len(selected_mm_ids) >= 6:
                break
            if cid not in selected:
                selected_mm_ids.append(cid)
                selected.append(cid)

        for cid in text_sorted:
            if len(selected) >= 20:
                break
            if cid not in selected:
                selected_text_ids.append(cid)
                selected.append(cid)

        final_selected = selected[:20]
        final_anchor_ids = [cid for cid in selected_anchor_ids if cid in final_selected]
        final_mm_ids = [cid for cid in selected_mm_ids if cid in final_selected]
        final_text_ids = [cid for cid in selected_text_ids if cid in final_selected]

        print("选中 chunk 总数（按顺序，最多10）：", len(final_selected))
        print("选中 chunk id（按顺序）：", final_selected)

        # 选中 chunk 的页码（若 chunk id 为 (page, id) 之类的结构）
        try:
            selected_pages = sorted({p for (p, _) in final_selected})
        except Exception:
            selected_pages = []
        print("选中 chunk 的页码（去重、排序）：", selected_pages)

        # 加载 chunk 内容
        chunks_anchor = _load_chunks(content_dir, pdf_name, text_chunkID=[], special_chunkID=final_anchor_ids)
        chunks_local = _load_chunks(content_dir, pdf_name, text_chunkID=final_text_ids, special_chunkID=final_mm_ids)

        chunks_merged = {
            "text": {**chunks_anchor.get("text", {}), **chunks_local.get("text", {})},
            "multimodal": {**chunks_anchor.get("multimodal", {}), **chunks_local.get("multimodal", {})}
        }

        # 生成回答
        if len(chunks_merged["multimodal"]) == 0:
            modal_info = "text"
            image_list = None
            prompt, response = unimodal_generator(query=query, answer=expected_answer, chunks=chunks_merged, chatLLM=chatLLM, save_dir=str(output_root))
        else:
            modal_info = "multimodal"
            ocr_imagefile_dir = ocr_cache_root / pdf_name / "auto"
            prompt, image_list, response = multimodal_generator(query=query, answer=expected_answer, chunks=chunks_merged, chatVLM=chatVLM, ocr_imagefile_dir=ocr_imagefile_dir, save_dir=str(output_root))

        # 判断与评分
        judgment = judge_answer(model=model, reasoning_model=reasoning_model, question=question, reference_answer=expected_answer, response=response, api_key=api_key, base_url=base_url)
        print(f"Judge 返回: {judgment}")

        is_correct = False
        if isinstance(judgment, str) and judgment.lower() in ("true", "false"):
            is_correct = (judgment.lower() == "true")

        score = score_answer(model=model, reasoning_model=reasoning_model, question=question, reference_answer=expected_answer, response=response, api_key=api_key, base_url=base_url)
        try:
            score_value = float(score)
            score_value = max(0.0, min(100.0, score_value))
        except Exception:
            score_value = 0.0

        # 在控制台清晰展示关键字段
        display_result(document_name=pdf_name, question=question, expected=expected_answer, response=response, judgment=judgment, score=score_value, modal=modal_info)

    finally:
        # 确保释放分区并关闭连接
        try:
            joint.milvus.release_partition(pdf_name)
        except Exception:
            pass
        joint.close()


if __name__ == "__main__":
    main()
