#根据不同的数据集格式（slidevqa / tatdqa / mmlongbench），统一解析成一个字典结构，并统计总共有多少条问答。

import json
from typing import Dict, List, Tuple, Any, Literal

Mode = Literal["slidevqa", "tatdqa", "mmlongbench"]

def load_qa_by_mode(json_path: str, mode: Mode) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """
    读取不同模式的QA数据，返回 doc_qa_dict 和总的 QA 数量。
    
    doc_qa_dict: {
        <doc_key>: [
            {"question": <q>, "answer": <a>},
            ...
        ],
        ...
    }
    - slidevqa:    doc_key 使用 item["deck_name"]
    - tatdqa:      doc_key 使用 item["doc"]["uid"]（与您原始代码保持一致）
    - mmlongbench: doc_key 使用 item["doc_id"]
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
    }
    mode = mode.lower()
    if mode not in parsers:
        raise ValueError(f"不支持的 mode：{mode}，可选：'slidevqa' | 'tatdqa' | 'mmlongbench'")

    doc_qa_dict = parsers[mode](data)

    # 3) 统计
    total_qa_count = sum(len(v) for v in doc_qa_dict.values())
    return doc_qa_dict, total_qa_count


def _parse_slidevqa(data: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
    """
    SlideVQA 格式示例：
    [
      {
        "deck_name": "...",
        "question": "...",
        "answer": "...",
        ...
      },
      ...
    ]
    """
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        doc_key = item.get("deck_name")
        if not doc_key:
            # 跳过异常项
            continue
        q = item.get("question")
        a = item.get("answer")
        doc_qa_dict.setdefault(doc_key, []).append({"question": q, "answer": a})
    return doc_qa_dict


def _parse_tatdqa(data: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
    """
    TATDQA（如 tatdqa_dataset_test_gold.json）格式示例：
    [
      {
        "doc": {"uid": "...", ...},
        "questions": [
          {"question": "...", "answer": [...] 或 数值/字符串, ...},
          ...
        ]
      },
      ...
    ]
    这里与您原始代码保持一致：doc_key 使用 doc["uid"]。
    """
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for doc_questions in data:
        doc = doc_questions.get("doc", {})
        doc_key = doc.get("uid")  # 与您原逻辑一致
        if not doc_key:
            # 如果想改为用文件名：doc_key = doc.get("source") or doc.get("uid")
            continue
        qs = doc_questions.get("questions", [])
        bucket = doc_qa_dict.setdefault(doc_key, [])
        for qa in qs:
            q = qa.get("question")
            a = qa.get("answer")
            bucket.append({"question": q, "answer": a})
    return doc_qa_dict


def _parse_mmlongbench(data: List[dict]) -> Dict[str, List[Dict[str, Any]]]:
    """
    MMLongBench-Doc_samples.json 格式示例：
    [
      {
        "doc_id": "...",
        "question": "...",
        "answer": "...",
        ...
      },
      ...
    ]
    """
    doc_qa_dict: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        doc_key = item.get("doc_id")
        if not doc_key:
            continue
        q = item.get("question")
        a = item.get("answer")
        doc_qa_dict.setdefault(doc_key, []).append({"question": q, "answer": a})
    return doc_qa_dict
