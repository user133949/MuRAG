""" 对某个 QA 系统（比如 RAG）生成的结果文件做自动化评测，
统计它的正确率、平均得分，并把详细评测结果保存成一个 CSV 文件。 """

import os
import sys
import json
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from Utils.LLMclient import ChatModel
from Utils.LLMJudge import judge_answer, score_answer
from Utils.LoggerUtil import get_logger

def evaluate_results(json_path1: str, dataset_name: str, judge_dir: str):
    ############################################# 系统配置 ###############################################
    # 指定显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # 配置日志
    log_dir = Path(__file__).parent / "Logs"    # 创建日志目录：当前文件同级的 logs 文件夹
    logger = get_logger(log_dir=log_dir, backup_count=10)
    logger.info("\n------ 系统启动 ------\n")

    ########################################### 大模型API配置 ############################################
    """阿里云 DashScope QWen 模型配置"""
    max_token_count = 32000
    model="qwen-plus" 
    tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
    reasoning_model = False
    embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    vl_model = "qwen-vl-plus"
    api_key="sk-xxx"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"

    ############################################## 路径设置 ###############################################
    # 选择数据集
    #dataset_name = "TAT-DQA"  # 可以改成 "TAT-DQA" 或 "MMLongBench-Doc" 或 "SlideVQA"
    # RAG 评价结果输出目录
    judge_dir = os.path.join(judge_dir, f"{dataset_name}_judge.csv")

    # 确保父目录存在
    judge_path = Path(judge_dir)
    judge_path.parent.mkdir(parents=True, exist_ok=True)

    ######################################################################################################
    json_path = Path(json_path1)
    if not json_path.exists():
        raise FileNotFoundError(f"结果文件不存在：{json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        NavieRAG_QA = json.load(f)
    print(f"--- NavieRAG 结果载入，共包含 {len(NavieRAG_QA)} 个QA ---")

    total_count = 0       # 总计数
    correct_count = 0     # 正确数
    score_sum = 0.0       # 总分数
    # 初始化结果统计 DataFrame
    results_df = pd.DataFrame(columns=["question", "answer", "response", "judgment", "score"])

    for qa in NavieRAG_QA:
        question = qa['query']
        response = qa['response']
        answer = qa['answer']
        
        # 1) 正确性判断（异常时记错误并用占位值）
        try:
            judgment = judge_answer(
                model=model,
                reasoning_model=reasoning_model,
                question=question,
                reference_answer=answer,
                response=response,
                api_key=api_key,
                base_url=base_url
            )
            print(judgment)
        except Exception as e:
            judgment = f"[ERROR] judge_answer: {e}"

        j = str(judgment).strip().lower()
        if j in ("true", "false") and j == "true":
            correct_count += 1
        total_count += 1

        # 2) 打分（异常时记错误并记为 0.0 分）
        try:
            score = score_answer(
                model=model,
                reasoning_model=reasoning_model,
                question=question,
                reference_answer=answer,
                response=response,
                api_key=api_key,
                base_url=base_url
            )
        except Exception as e:
            score = f"[ERROR] score_answer: {e}"
        print(score)

        try:
            score_value = float(score)
        except ValueError:
            score_value = 0.0
        score_sum += score_value

        results_df.loc[len(results_df)] = [question, answer, response, judgment, score]
        
        print(f"第 {total_count} 个 QA 完成")
        logger.info(f"\n------ 第 {total_count} 个 QA 完成 ------\n")

    # 计算准确率和平均得分
    accuracy = correct_count / total_count if total_count else 0.0
    average_score = score_sum / total_count if total_count else 0.0
    logger.info(f"total_count: {total_count}")
    logger.info(f"correct_count: {correct_count}, Accuracy: {accuracy}")
    logger.info(f"score_sum: {score_sum}, Average Score: {average_score}")

    # 保存结果到 CSV 文件
    results_df.to_csv(judge_dir, index=False, encoding="utf-8-sig")
    return accuracy, average_score, str(judge_path)