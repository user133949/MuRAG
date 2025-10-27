import os
import sys
import json
import pandas as pd
from sentence_transformers import SentenceTransformer 
from pathlib import Path

# 添加父目录到路径，确保可以导入相关模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from LLM.LLMclient import ChatModel
from DocProcess.DocumentProcessor import parsed_document_process
from GraphProcess.EntityDisambiguation import run_disambiguation_pipeline
from GraphProcess.GraphVectorProcessor import process_vectors_and_clean_graph

class GraphProcessor:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化图谱处理器
        
        Args:
            api_key: 阿里云API密钥
            base_url: API基础URL，默认为DashScope
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # 获取根目录（当前文件的父目录的父目录）
        self.root_dir = Path(__file__).resolve().parent.parent
        
        # 模型配置
        self.max_token_count = 32000
        self.model = "qwen-plus" 
        self.tokenizer = "Qwen/Qwen3-4B-Instruct-2507"
        self.reasoning_model = False
        self.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
        self.vl_model = "qwen-vl-plus-2025-08-15"
        
        # 初始化模型客户端
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化LLM模型和编码器"""
        try:
            self.chatLLM = ChatModel(
                model=self.model,
                reasoning_model=self.reasoning_model, 
                api_key=self.api_key, 
                base_url=self.base_url
            )
            self.chatVLM = ChatModel(
                model=self.vl_model, 
                reasoning_model=False, 
                api_key=self.api_key, 
                base_url=self.base_url
            )
            self.encoder = SentenceTransformer(self.embedding_model)
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
    
    def contentlist_to_graph(self, pdf_name, similarity_threshold=0.95, json_mode=True):
        """
        处理PDF文档并生成知识图谱
        
        Args:
            pdf_name: PDF文档名称（不含扩展名）
            similarity_threshold: 实体消歧相似度阈值
            json_mode: 是否使用JSON格式的三元组提取
            
        Returns:
            dict: 包含处理结果的状态和信息
        """
        try:
            # 构建路径 - data目录在根目录下
            data_dir = self.root_dir / "data"
            graph_dir = data_dir / "Graph"
            
            # OCR相关路径
            cache_dir = data_dir / "parsed" / pdf_name / "auto"
            # ocr_dir = data_dir / "parsed"
            
            # 检查最终图谱是否已存在
            final_graph_path = graph_dir / "FinalGraph" / f"{pdf_name}_final_graph.json"
            if final_graph_path.exists():
                print("检测到已建图")
                return {
                    "status": "skipped",
                    "message": f"最终图谱已存在: {final_graph_path}",
                    "pdf_name": pdf_name,
                    "final_graph_path": str(final_graph_path)
                }
            
            print(f"开始处理PDF: {pdf_name}")
            
            # 文档内容读取与建图
            graph = parsed_document_process(
                pdf_name=pdf_name, 
                json_file_path=cache_dir, 
                image_file_dir=cache_dir, 
                chatLLM=self.chatLLM, 
                chatVLM=self.chatVLM,
                max_token_count=self.max_token_count, 
                model=self.model, 
                encoding_model=self.tokenizer, 
                encoder=self.encoder, 
                json_mode=json_mode
            )
            
            print(f"知识图谱初步构建完成，共包含 {len(graph)} 个元素")
            
            # 保存原始图谱
            origin_graph_dir = graph_dir / "OriginalGraph"
            origin_graph_dir.mkdir(parents=True, exist_ok=True)
            origin_graph_path = origin_graph_dir / f"{pdf_name}_original_graph.json"
            
            with open(origin_graph_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=4, ensure_ascii=False)
            
            print(f"原始知识图谱保存为: {origin_graph_path}")
            
            # 实体消歧
            final_graph_with_vectors = run_disambiguation_pipeline(
                graph, 
                similarity_threshold=similarity_threshold
            )
            
            # 清理图谱向量
            final_graph = process_vectors_and_clean_graph(
                final_graph_with_vectors=final_graph_with_vectors
            )
            
            # 保存最终图谱
            final_graph_dir = graph_dir / "FinalGraph"
            final_graph_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存带向量的图谱
            final_graph_with_vector_path = final_graph_dir / f"{pdf_name}_final_graph_with_vector.json"
            with open(final_graph_with_vector_path, 'w', encoding='utf-8') as f:
                json.dump(final_graph_with_vectors, f, indent=4, ensure_ascii=False)
            
            # 保存最终图谱
            final_graph_path = final_graph_dir / f"{pdf_name}_final_graph.json"
            with open(final_graph_path, 'w', encoding='utf-8') as f:
                json.dump(final_graph, f, indent=4, ensure_ascii=False)
            
            print(f"最终知识图谱保存为: {final_graph_path}")
            
            return {
                "status": "success",
                "message": "知识图谱生成完成",
                "pdf_name": pdf_name,
                "original_graph_path": str(origin_graph_path),
                "final_graph_path": str(final_graph_path),
                "final_graph_with_vector_path": str(final_graph_with_vector_path),
                "graph_stats": {
                    "original_elements": len(graph),
                    "final_elements": len(final_graph)
                }
            }
            
        except Exception as e:
            error_msg = f"处理PDF {pdf_name} 时发生错误: {str(e)}"
            print(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "pdf_name": pdf_name
            }

# 使用示例
if __name__ == "__main__":
    # 示例用法
    processor = GraphProcessor(api_key="xxx")
    
    # 处理单个PDF
    result = processor.process_pdf_to_graph(
        pdf_name="example_document"
    )
    print(result)