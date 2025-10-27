import json
from typing import Dict, List, Tuple, Any, Literal
import ast

Mode = Literal[
    "slidevqa",
    "tatdqa",
    "mmlongbench",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
    "financebench",  # 新增
]


def load_qa_by_mode(json_path: str, mode: Mode) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """
    读取不同模式的QA数据，返回 doc_qa_dict 和总的 QA 数量。
    """
    # 1) 读取 JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 文件不存在：{json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{json_path}，错误：{e}")

    # 2) 分模式解析
    parsers = {
        "slidevqa": _parse_slidevqa,
        "tatdqa": _parse_tatdqa,
        "mmlongbench": _parse_mmlongbench,
        "shiftproject_test": _parse_generic_synthetic,
        "syntheticDocQA_artificial_intelligence_test": _parse_generic_synthetic,
        "syntheticDocQA_energy_test": _parse_generic_synthetic,
        "syntheticDocQA_government_reports_test": _parse_generic_synthetic,
        "syntheticDocQA_healthcare_industry_test": _parse_generic_synthetic,
        "financebench": _parse_financebench,  # 新增
    }

    if mode not in parsers:
        raise ValueError(f"不支持的 mode：{mode}")

    doc_qa_dict = parsers[mode](data, mode)

    # 3) 统计
    total_qa_count = sum(len(v) for v in doc_qa_dict.values())
    return doc_qa_dict, total_qa_count


# ========== 各种解析函数 ==========

def _parse_slidevqa(data: List[dict], mode: str = "slidevqa") -> Dict[str, List[Dict[str, Any]]]:
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        doc_key = item.get("deck_name")
        if not doc_key:
            continue
        qa_item = dict(item)
        # 修正这里：使用 "evidence_pages" 而不是 "evidence_page"
        qa_item["evidence_page"] = item.get("evidence_pages", [])  # 注意这里的修正
        if not isinstance(qa_item["evidence_page"], list):
            qa_item["evidence_page"] = [qa_item["evidence_page"]]
        doc_key = doc_key.replace(".pdf", "")
        doc_qa_dict.setdefault(doc_key, []).append(qa_item)
    return doc_qa_dict


def _parse_tatdqa(data: List[dict], mode: str = "tatdqa") -> Dict[str, List[Dict[str, Any]]]:
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for doc_questions in data:
        doc = doc_questions.get("doc", {})
        doc_key = doc.get("uid")
        if not doc_key:
            continue
        doc_key = doc_key.replace(".pdf", "")
        bucket = doc_qa_dict.setdefault(doc_key, [])
        for qa in doc_questions.get("questions", []):
            qa_item = dict(qa)
            qa_item.pop("evidence_page", None)  # 确保没有 evidence_page
            bucket.append(qa_item)
    return doc_qa_dict


def _parse_mmlongbench(data: List[dict], mode: str = "mmlongbench") -> Dict[str, List[Dict[str, Any]]]:
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        doc_key = item.get("doc_id", "")
        if not doc_key:
            continue
        doc_key = doc_key.replace(".pdf", "")
        qa_item = dict(item)

        # evidence_pages 可能是字符串
        try:
            raw_pages = item.get("evidence_pages", "[]")
            if isinstance(raw_pages, str):
                qa_item["evidence_page"] = [int(p) for p in ast.literal_eval(raw_pages)]
            else:
                qa_item["evidence_page"] = [int(p) for p in raw_pages]
        except Exception:
            qa_item["evidence_page"] = []

        # evidence_sources 是字符串的 Python list
        try:
            raw_sources = item.get("evidence_sources", "[]")
            if isinstance(raw_sources, str):
                qa_item["evidence_sources"] = ast.literal_eval(raw_sources)
            else:
                qa_item["evidence_sources"] = raw_sources
        except Exception:
            qa_item["evidence_sources"] = []

        doc_qa_dict.setdefault(doc_key, []).append(qa_item)
    return doc_qa_dict


def _parse_generic_synthetic(data: List[dict], mode: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    用于处理以下五个格式相同的数据集：
    shiftproject_test
    syntheticDocQA_artificial_intelligence_test
    syntheticDocQA_energy_test
    syntheticDocQA_government_reports_test
    syntheticDocQA_healthcare_industry_test

    格式：
    [
      {
        "query": "...",
        "answer": "...",
        "evidence_pages": ["1"]
      },
      ...
    ]

    修改：跳过 query 为 null 或空值的噪声数据
    """
    doc_key = mode  # 每个数据集 doc_key 即数据集名称
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {doc_key: []}

    for item in data:
        # 跳过 query 为 null 或空值的噪声数据
        query = item.get("query")
        if query is None or query == "":
            continue

        qa_item = {
            "question": query,
            "answer": item.get("answer"),
            "evidence_page": [str(x) for x in item.get("evidence_pages", [])],
        }
        doc_qa_dict[doc_key].append(qa_item)
    return doc_qa_dict


def _parse_financebench(data: List[dict], mode: str = "financebench") -> Dict[str, List[Dict[str, Any]]]:
    """
    FinanceBench 数据集格式解析：
    - doc_name 作为 doc_key
    - question / answer 字段直接取
    - evidence_page 从 evidence 列表里的 evidence_page_num 提取
    - 其他字段展平
    """
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}

    for item in data:
        doc_key = item.get("doc_name")
        if not doc_key:
            continue
        doc_key = doc_key.replace(".pdf", "")

        qa_item = dict(item)  # 先展平所有字段
        qa_item["pdf_name"] = item.get("doc_name")
        qa_item["question"] = item.get("question")
        qa_item["answer"] = item.get("answer")

        # 解析 evidence_page
        evidence_list = item.get("evidence", [])
        if isinstance(evidence_list, list):
            pages = []
            for ev in evidence_list:
                page = ev.get("evidence_page_num")
                if page is not None:
                    pages.append(page)
            qa_item["evidence_page"] = pages
        else:
            qa_item["evidence_page"] = []

        doc_qa_dict.setdefault(doc_key, []).append(qa_item)

    return doc_qa_dict
