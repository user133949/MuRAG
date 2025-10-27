from pathlib import Path
import json
from typing import Union, List, Dict, Optional

def print_sources(resp, top_k: int = 5):
    """打印本轮回答命中的来源片段与元信息。"""
    srcs = getattr(resp, "source_nodes", None) or []
    if not srcs:
        print("\n[提示] 本轮未返回来源片段。")
        return
    print("\n—— 命中来源 ——")
    for i, node in enumerate(srcs[:top_k], 1):
        meta = node.metadata or {}
        file_path = meta.get("file_path") or meta.get("filename") or meta.get("file_name") or "N/A"
        try:
            file_name = Path(file_path).name
        except Exception:
            file_name = file_path
        score = getattr(node, "score", None)
        score_str = f"{score:.4f}" if isinstance(score, (float, int)) else "N/A"
        text = (node.get_text() if hasattr(node, "get_text") else getattr(node, "text", "")) or ""
        text = text.replace("\n", " ").strip()
        if len(text) > 280:
            text = text[:280] + "..."
        page = meta.get("page_label") or meta.get("page_number")
        page_info = f" | 页: {page}" if page is not None else ""
        print(f"[{i}] 文件: {file_name}{page_info} | 相似度: {score_str}\n    片段: {text}")
    print("———————\n")


def uid_from_txt_name(p: Path) -> str:
    """ 0ac9...cbaf_origin.txt -> 0ac9...cbaf """
    name = p.stem
    if name.endswith("_origin"):
        return name[:-7]  # 去掉 "_origin"
    elif name.endswith("_content_list"):
        return name[:-13]  # 去掉 "_content_list"
    else:
        return name  # 默认返回文件名（无扩展名）


def save_qa_list(save_path: Union[str, Path], qa_records: List[Dict]) -> None:
    """
    将多个 QA 结果保存为一个 JSON 列表文件。

    参数：
      - save_path: 输出文件路径（.json）
      - qa_records: 列表，每个元素是一个 dict，结构必须包含：
          {
            "query": str,
            "response": str,
            "answer": List[str]
          }
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_path.write_text(
        json.dumps(qa_records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[保存成功] {len(qa_records)} 条 QA 已保存到 {save_path}")


def clean_and_extract_core_entities(raw_entities: list) -> list:
    """
    清理和提取核心实体。

    该函数会执行以下操作：
    1. 从每个实体中移除预设的停用词。
    2. 标准化所有空白字符（如 \n, \t, 多余空格）为单个空格。
    3. 将所有实体转换为大写。
    4. 移除重复的实体和空字符串。

    Args:
        raw_entities (list): 包含原始实体字符串的列表。

    Returns:
        list: 包含清理、去重后的大写核心实体的列表。
    """
    stopwords = ["的", "之间", "是", "什么", "关系"]
    core_entities = set()

    for entity in raw_entities:
        cleaned_entity = entity

        # 步骤 1: 移除停用词
        for stopword in stopwords:
            cleaned_entity = cleaned_entity.replace(stopword, "")

        # 步骤 2: 标准化空白字符
        cleaned_entity = " ".join(cleaned_entity.split())

        # 步骤 3: 转换为大写
        cleaned_entity = cleaned_entity.upper()

        # 步骤 4: 如果处理后实体不为空，则添加到集合中（自动去重）
        if cleaned_entity:
            core_entities.add(cleaned_entity)

    # 将集合转换为列表并返回
    return list(core_entities)
