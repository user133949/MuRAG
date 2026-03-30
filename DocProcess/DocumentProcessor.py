import os
import json
import logging
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from LLM.LLMclient import ChatModel
from EntityExtraction.DomainDetector import generate_domain
from EntityExtraction.LanguageDetector import detect_language
from EntityExtraction.PersonaGenerator import generate_persona
from EntityExtraction.EntityTypesGenerator import generate_entity_types
from EntityExtraction.EntityRelationExtract import entity_relationship_extraction
from EntityExtraction.PromptGenerator import generate_entity_relationship_examples
from EntityExtraction.PromptGenerator import create_extract_graph_prompt
from EntityExtraction.PromptGenerator import prompt_concatenate
from DocProcess.RelationshipSupplement import create_id_and_embedding
from DocProcess.RelationshipSupplement import create_anchor_node
from DocProcess.RelationshipSupplement import create_anchor_edge
from DocProcess.RelationshipSupplement import creat_image_node
from DocProcess.RelationshipSupplement import creat_image_edges
from DocProcess.RelationshipSupplement import creat_equation_node
from DocProcess.RelationshipSupplement import creat_table_node

def _document_loader(file_path: Path, 
                     pdf_name: str
                     )-> tuple[dict, str]:
    """
    读取指定文件夹中最新的 OCR 解析文件，并按页码分组
    
    参数:
    file_path: JSON 文件的文件夹路径
    pdf_name: 文档名称

    返回:
        按页码分组的字典
        pages = {
            0: ['段落1', '段落2'],
            1: ['段落1'],
            2: ['段落1', '段落2', '段落3']
            key: value
        }
        latest_file: 最新的 JSON 文件路径
    """
    json_file_path = os.path.join(file_path, f"{pdf_name}_content_list.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages = defaultdict(list)
    for item in data:
        pages[item['page_idx']].append(item)

    return pages, str(json_file_path)

def parsed_document_process(pdf_name: str,
                            json_file_path: Path,
                            image_file_dir: Path, 
                            chatLLM: ChatModel, 
                            chatVLM: ChatModel, 
                            max_token_count: int,
                            model: str,
                            encoding_model: str,
                            encoder: SentenceTransformer,
                            json_mode: bool | None=True, 
                            output_path: Path | None = None
                            )-> list:
    """
    处理OCR解析后的文档 生成文档的多模态知识图谱

    参数:
        pdf_name: 文档名称
        json_file_path: 文档解析后的 JSON 文件路径
        image_file_dir: 图片文件的文件夹路径
        chatLLM: 调用的 LLM 模型
        chatVLM: 调用的 VLM 模型
        max_token_count: 最大 token 数量
        model: 模型名称
        encoding_model: 编码模型名称，用于分词器
        encoder: 向量编码器，用于生成语义向量
        json_mode: 是否使用 JSON 格式
        output_path: prompt_template的保存路径

    返回:
        文档的多模态知识图谱
    """
    # 读取文档并按页码分组
    pages, latest_file = _document_loader(file_path=json_file_path, pdf_name=pdf_name)
    if pages:
        print("{} 文档加载成功，共加载了 {} 页".format(latest_file, len(pages)))
        logging.info("{} 文档加载成功，共加载了 {} 页".format(latest_file, len(pages)))

        # 提取第一个非空页和前三个非空页的 "text" 字段内容
        text_list_1stPage = []
        text_list_3Page = []

        non_empty_pages = []  # 保存非空页的内容
        for page_num, items in sorted(pages.items()):
            text_list_page = [item["text"] for item in items if "text" in item and item["text"].strip()]
            if text_list_page:  # 非空才加入
                non_empty_pages.append((page_num, text_list_page))
                
        # 取第一个非空页
        if non_empty_pages:
            text_list_1stPage = non_empty_pages[0]
        else:
            logging.error("文档没有任何可处理的文本内容")
            return []

        # 取前三个非空页
        for _, page_content in non_empty_pages[:3]:
            text_list_3Page.extend(page_content)

        if not text_list_3Page:
            logging.error("前三个非空页没有可处理的文本内容")
            return []

        # 用前三页内容，提取 领域 语言，并为LLM设定系统角色
        domain = generate_domain(chat_model=chatLLM, docs=text_list_3Page)
        language = detect_language(chat_model=chatLLM, docs=text_list_3Page)
        persona = generate_persona(chat_model=chatLLM, domain=domain)
        # 用前三页内容，生成实体类型
        entity_types = generate_entity_types(chat_model=chatLLM, 
                                             domain=domain, 
                                             persona=persona, 
                                             docs=text_list_3Page)
        # 用首页内容，生成实体关系抽取例子，合成 prompt 模板
        examples = generate_entity_relationship_examples(chat_model=chatLLM, 
                                                         entity_types=entity_types, 
                                                         docs=text_list_1stPage, 
                                                         language=language, 
                                                         json_mode=json_mode, 
                                                         persona=persona)
        prompt_template = create_extract_graph_prompt(entity_types=entity_types, 
                                                      docs=text_list_1stPage, 
                                                      examples=examples,
                                                      language=language,
                                                      max_token_count=max_token_count,
                                                      model=model,
                                                      encoding_model=encoding_model,
                                                      json_mode=json_mode,
                                                      output_path=output_path,
                                                      min_examples_required=1, 
                                                      max_examples_allowed=2)
        logging.info(f"persona: {persona}")
        logging.info(f"prompt_template: {prompt_template}")

        graph = []              # 文档级图谱初始化
        anchor_node_last = []   # 段落锚节点初始化
        for page_num, items in sorted(pages.items()):

            print(f"\n=== 处理第 {page_num} 页 ===")
            logging.info(f"\n=== 处理第 {page_num} 页 ===")
            
            # 每页从第1个段落开始编号
            segment_idx = 0         # 段落计数器
            subgraph_page = []      # 页面级子图初始化
            for item in items:
                subgraph_seg = []   # 段落级子图初始化

                item_type = item.get('type')
                if item_type in ['text', 'image', 'equation', 'table']:
                    if item_type == 'text':
                        # 处理 text 类型
                        text_str = item.get('text', '') # 没有这个键时，返回空字符串 ''
                        page_idx = item.get('page_idx', '')
                        text = [text_str]
                        if text_str.strip() == '':
                            print("空文本段落，跳过")
                            segment_idx += 1
                            continue
                        text_triplet = []
                        # 1.LLM提取三元组
                        prompt = prompt_concatenate(prompt=prompt_template, docs=text)
                        extraction_result = entity_relationship_extraction(chat_model=chatLLM, 
                                                                           persona=persona, 
                                                                           extract_prompt=prompt, 
                                                                           max_retries=10)
                        # 2.如果抽取成功，给每个都附上 ID，成为段落级子图
                        if extraction_result:
                            text_triplet = create_id_and_embedding(extraction_result, page_idx, segment_idx, 
                                                                   pdf_name, encoder)
                        else:
                            logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                            print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                        subgraph_seg.extend(text_triplet)
                        text_list = text
                        
                    elif item_type == 'image':
                        # 处理 image 类型
                        img_path = item.get('img_path', '')
                        img_caption = item.get('image_caption', [])
                        img_footnote = item.get('image_footnote', [])
                        page_idx = item.get('page_idx', '')

                        image_triplet = []
                        image_node = []
                        image_edge = []
                        # 1.处理文本内容
                        merged_text = []
                        if isinstance(img_caption, list):
                            merged_text.extend(img_caption)
                        if isinstance(img_footnote, list):
                            merged_text.extend(img_footnote)
                        if merged_text:
                            # 标题和脚注不为空，生成三元组
                            prompt = prompt_concatenate(prompt=prompt_template, docs=merged_text)
                            extraction_result = entity_relationship_extraction(chat_model=chatLLM,
                                                                                persona=persona,
                                                                                extract_prompt=prompt,
                                                                                max_retries=10)
                            if extraction_result:
                                # 附上 ID 成为图片三元组
                                image_triplet = create_id_and_embedding(extraction_result, page_idx, segment_idx, 
                                                                        pdf_name, encoder)
                            else:
                                logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                                print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                        # 2.处理原图 节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            image_node, image_description = creat_image_node(chatVLM, pdf_name, image_file_dir, img_path, item_type, 
                                                                             merged_text, page_idx, segment_idx, encoder)
                        else:
                            image_node = []
                            image_description = ""
                        # 3.处理原图节点 与 image 段落三元组 边
                        if image_triplet and image_node:
                            image_edge = creat_image_edges(pdf_name, image_node, image_triplet, 5)

                        # 最终的图片段落的子图
                        subgraph_seg.extend(image_triplet)
                        subgraph_seg.extend(image_node)
                        subgraph_seg.extend(image_edge)
                        text_list = merged_text + ([image_description] if image_description else [])


                    elif item_type == 'equation':
                        # 处理 equation 类型
                        img_path = item.get('img_path', '')
                        text = item.get('text', '')
                        page_idx = item.get('page_idx', '')

                        equation_text_node = []
                        equation_image_node = []
                        equation_image_description = ""

                        # 1.处理公式的文本内容
                        if text:
                            equation_text_node = creat_equation_node(pdf_name, text, page_idx, segment_idx, encoder)
                        # 2.处理公式的原图节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            equation_image_node, equation_image_description = creat_image_node(
                                chatVLM, pdf_name, image_file_dir, img_path, item_type,
                                text, page_idx, segment_idx, encoder
                            )

                        # 3.合并子图
                        subgraph_seg.extend(equation_text_node)
                        subgraph_seg.extend(equation_image_node)

                        text_list = [text] if text else []
                        if equation_image_description:
                            text_list.append(equation_image_description)


                    elif item_type == 'table':
                        # 处理 table 类型
                        img_path = item.get('img_path', '')
                        table_caption = item.get('table_caption', '')
                        table_footnote = item.get('table_footnote', '')
                        table_body = item.get('table_body', [])
                        page_idx = item.get('page_idx', '')

                        table_triplet = []
                        table_text_node = []
                        table_image_node = []
                        table_edge1 = []
                        table_edge2 = []
                        table_edge3 = []
                        table_image_description = ""

                        # 1.处理表格标题脚注文本内容
                        merged_text = []
                        if isinstance(table_caption, list):
                            merged_text += table_caption
                        if isinstance(table_footnote, list):
                            merged_text += table_footnote
                        if merged_text:
                            prompt = prompt_concatenate(prompt=prompt_template, docs=merged_text)
                            extraction_result = entity_relationship_extraction(
                                chat_model=chatLLM,
                                persona=persona,
                                extract_prompt=prompt,
                                max_retries=10
                            )
                            if extraction_result:
                                table_triplet = create_id_and_embedding(
                                    extraction_result, page_idx, segment_idx, pdf_name, encoder
                                )
                            else:
                                logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                                print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")

                        # 2.处理表格文本内容
                        if table_body:
                            table_text_node = creat_table_node(pdf_name, table_body, page_idx, segment_idx, encoder)

                        # 3.处理表格原图 节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            table_image_node, table_image_description = creat_image_node(
                                chatVLM, pdf_name, image_file_dir, img_path, item_type,
                                merged_text, page_idx, segment_idx, encoder
                            )

                        # 4.处理原图节点 与 表格标题脚注 段落三元组 边
                        if table_triplet and table_image_node:
                            table_edge1 = creat_image_edges(pdf_name, table_image_node, table_triplet, 5)
                        # 5.处理原图节点 与 表格文本内容 三元组 边
                        if table_text_node and table_image_node:
                            table_edge2 = creat_image_edges(pdf_name, table_image_node, table_text_node, 5)
                        # 6.表格文本内容 三元组 与 表格标题脚注 段落三元组 边
                        if table_triplet and table_text_node:
                            table_edge3 = creat_image_edges(pdf_name, table_text_node, table_triplet, 5)

                        # 7.合并表格段落的子图
                        subgraph_seg.extend(table_triplet)
                        subgraph_seg.extend(table_text_node)
                        subgraph_seg.extend(table_image_node)
                        subgraph_seg.extend(table_edge1)
                        subgraph_seg.extend(table_edge2)
                        subgraph_seg.extend(table_edge3)

                        text_list = merged_text + ([table_body] if table_body else [])
                        if table_image_description:
                            text_list.append(table_image_description)

                    # 创建本段的锚节点，与本段以及上一段的锚节点建立 anchor_edge
                    anchor_node = create_anchor_node(pdf_name=pdf_name, segment_text=text_list, page_idx=page_idx, 
                                                     segment_idx=segment_idx, chatLLM=chatLLM, encoder=encoder)
                    anchor_edge = create_anchor_edge(pdf_name=pdf_name, 
                                                     source_triplet=subgraph_seg, 
                                                     target_triplet=anchor_node, 
                                                     local=True)
                    if anchor_node_last:
                        anchor_edge_last = create_anchor_edge(pdf_name=pdf_name,
                                                              source_triplet=anchor_node, 
                                                              target_triplet=anchor_node_last, 
                                                              local=False)
                        subgraph_page.extend(subgraph_seg)
                        subgraph_page.extend(anchor_node)
                        subgraph_page.extend(anchor_edge)
                        subgraph_page.extend(anchor_edge_last)
                    else:
                        subgraph_page.extend(subgraph_seg)
                        subgraph_page.extend(anchor_node)
                        subgraph_page.extend(anchor_edge)
                    segment_idx += 1
                else:
                    print(f"未知类型: {item_type}")
                    segment_idx += 1
                
                anchor_node_last = anchor_node      # 缓存本次段落锚节点
            anchor_node_last = anchor_node      # 跨页时缓存段落锚节点

            # 页面级子图 归档到 文档级图 graph
            graph.extend(subgraph_page)

            # 每完成一页，保存一次文档级图，防止丢失
            script_dir = Path(__file__).parent.absolute()   # 获取当前脚本所在目录
            save_path_doc = script_dir / "json_llm_extract_result_doc.json"
            save_path_doc.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            with open(save_path_doc, "w", encoding="utf-8") as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)

        logging.info(f"原始图谱构建完成，共 {len(graph)} 个元素")
    
    else:
        graph = []
        print("Error: 路径不存在或没有可加载的文档")
        logging.error("路径不存在或没有可加载的文档")

    return graph

def parsed_document_process_recovery(pdf_name: str, 
                                     anchor_node_description: str,
                                     json_file_path: Path, 
                                     image_file_dir: Path, 
                                     chatLLM: ChatModel, 
                                     chatVLM: ChatModel, 
                                     encoder: SentenceTransformer, 
                                     persona: str | None = None,
                                     prompt_template: str | None = None,
                                     begin_page: int = 1, 
                                     last_segment_idx: int = 0,
                                     incomplete_graph: list = []
                                     )-> list:
    """
    恢复 处理OCR解析后的文档 生成文档的多模态知识图谱

    参数:
        json_file_path: 文档解析后的 JSON 文件路径，默认选择最新一个文件
        image_file_dir: 图片文件的文件夹路径
        chatLLM: 调用的 LLM 模型
        chatVLM: 调用的 VLM 模型
        max_token_count: 最大 token 数量
        model: 模型名称
        encoding_model: 编码模型名称，用于分词器
        encoder: 向量编码器，用于生成语义向量
        json_mode: 是否使用 JSON 格式
        output_path: prompt_template的保存路径
        
        begin_page: 开始恢复的页码
        last_segment_idx: 上一段的段落计数器
        incomplete_graph: 不完整的知识图谱

    返回:
        文档的多模态知识图谱
    """
    # 读取文档并按页码分组
    pages, latest_file = _document_loader(file_path=json_file_path)
    if pages:
        print("{} 文档加载成功，共加载了 {} 页".format(latest_file, len(pages)))
        logging.info("{} 文档加载成功，共加载了 {} 页".format(latest_file, len(pages)))
        
        persona = persona
        prompt_template = prompt_template

        logging.info(f"persona: {persona}")
        logging.info(f"prompt_template: {prompt_template}")

        graph = incomplete_graph        # 载入不完整的知识图谱
        anchor_node_last = [{
        "name": f"segment anchor node for page {begin_page-1}, segment {last_segment_idx}",
        "type": "SEGMENT ANCHOR NODE",
        "description": anchor_node_description,
        "entityID": str([begin_page-1, last_segment_idx, "anchor"]), 
        "chunkID": str([[begin_page-1, last_segment_idx]]), 
        "ref_doc_id": pdf_name
        }]

        for page_num, items in sorted(pages.items()):

            if page_num < begin_page:
                continue

            print(f"\n=== 处理第 {page_num} 页 ===")
            logging.info(f"\n=== 处理第 {page_num} 页 ===")
            
            # 每页从第1个段落开始编号
            segment_idx = 0         # 段落计数器
            subgraph_page = []      # 页面级子图初始化
            for item in items:
                subgraph_seg = []   # 段落级子图初始化

                item_type = item.get('type')
                if item_type in ['text', 'image', 'equation', 'table']:
                    if item_type == 'text':
                        # 处理 text 类型
                        text_str = item.get('text', '') # 没有这个键时，返回空字符串 ''
                        page_idx = item.get('page_idx', '')
                        text = [text_str]
                        if text_str.strip() == '':
                            print("空文本段落，跳过")
                            segment_idx += 1
                            continue
                        text_triplet = []
                        # 1.LLM提取三元组
                        prompt = prompt_concatenate(prompt=prompt_template, docs=text)
                        extraction_result = entity_relationship_extraction(chat_model=chatLLM, 
                                                                           persona=persona, 
                                                                           extract_prompt=prompt, 
                                                                           max_retries=10)
                        # 2.如果抽取成功，给每个都附上 ID，成为段落级子图
                        if extraction_result:
                            text_triplet = create_id_and_embedding(extraction_result, page_idx, segment_idx, 
                                                                   pdf_name, encoder)
                        else:
                            logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                            print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                        subgraph_seg.extend(text_triplet)
                        text_list = text
                        

                    elif item_type == 'image':
                        # 处理 image 类型
                        img_path = item.get('img_path', '')
                        img_caption = item.get('image_caption', [])
                        img_footnote = item.get('image_footnote', [])
                        page_idx = item.get('page_idx', '')

                        image_triplet = []
                        image_node = []
                        image_edge = []
                        # 1.处理文本内容
                        merged_text = []
                        if isinstance(img_caption, list):
                            merged_text.extend(img_caption)
                        if isinstance(img_footnote, list):
                            merged_text.extend(img_footnote)
                        if merged_text:
                            # 标题和脚注不为空，生成三元组
                            prompt = prompt_concatenate(prompt=prompt_template, docs=merged_text)
                            extraction_result = entity_relationship_extraction(chat_model=chatLLM,
                                                                                persona=persona,
                                                                                extract_prompt=prompt,
                                                                                max_retries=10)
                            if extraction_result:
                                # 附上 ID 成为图片三元组
                                image_triplet = create_id_and_embedding(extraction_result, page_idx, segment_idx, 
                                                                        pdf_name, encoder)
                            else:
                                logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                                print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                        # 2.处理原图 节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            image_node, image_description = creat_image_node(chatVLM, pdf_name, image_file_dir, img_path, item_type, 
                                                                             merged_text, page_idx, segment_idx, encoder)
                        else:
                            image_node = []
                            image_description = ""
                        # 3.处理原图节点 与 image 段落三元组 边
                        if image_triplet and image_node:
                            image_edge = creat_image_edges(pdf_name, image_node, image_triplet, 5)

                        # 最终的图片段落的子图
                        subgraph_seg.extend(image_triplet)
                        subgraph_seg.extend(image_node)
                        subgraph_seg.extend(image_edge)
                        text_list = merged_text + ([image_description] if image_description else [])


                    elif item_type == 'equation':
                        # 处理 equation 类型
                        img_path = item.get('img_path', '')
                        text = item.get('text', '')
                        page_idx = item.get('page_idx', '')

                        equation_text_node = []
                        equation_image_node = []
                        equation_image_description = ""

                        # 1.处理公式的文本内容
                        if text:
                            equation_text_node = creat_equation_node(pdf_name, text, page_idx, segment_idx, encoder)
                        # 2.处理公式的原图节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            equation_image_node, equation_image_description = creat_image_node(
                                chatVLM, pdf_name, image_file_dir, img_path, item_type,
                                text, page_idx, segment_idx, encoder
                            )

                        # 3.合并子图
                        subgraph_seg.extend(equation_text_node)
                        subgraph_seg.extend(equation_image_node)

                        text_list = [text] if text else []
                        if equation_image_description:
                            text_list.append(equation_image_description)


                    elif item_type == 'table':
                        # 处理 table 类型
                        img_path = item.get('img_path', '')
                        table_caption = item.get('table_caption', '')
                        table_footnote = item.get('table_footnote', '')
                        table_body = item.get('table_body', [])
                        page_idx = item.get('page_idx', '')

                        table_triplet = []
                        table_text_node = []
                        table_image_node = []
                        table_edge1 = []
                        table_edge2 = []
                        table_edge3 = []
                        table_image_description = ""

                        # 1.处理表格标题脚注文本内容
                        merged_text = []
                        if isinstance(table_caption, list):
                            merged_text += table_caption
                        if isinstance(table_footnote, list):
                            merged_text += table_footnote
                        if merged_text:
                            prompt = prompt_concatenate(prompt=prompt_template, docs=merged_text)
                            extraction_result = entity_relationship_extraction(
                                chat_model=chatLLM,
                                persona=persona,
                                extract_prompt=prompt,
                                max_retries=10
                            )
                            if extraction_result:
                                table_triplet = create_id_and_embedding(
                                    extraction_result, page_idx, segment_idx, pdf_name, encoder
                                )
                            else:
                                logging.error(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")
                                print(f"page {page_idx} paragraph {segment_idx} 实体关系抽取失败")

                        # 2.处理表格文本内容
                        if table_body:
                            table_text_node = creat_table_node(pdf_name, table_body, page_idx, segment_idx, encoder)

                        # 3.处理表格原图 节点
                        if img_path and img_path.strip():   # ✅ 增加 img_path 判空
                            table_image_node, table_image_description = creat_image_node(
                                chatVLM, pdf_name, image_file_dir, img_path, item_type,
                                merged_text, page_idx, segment_idx, encoder
                            )

                        # 4.处理原图节点 与 表格标题脚注 段落三元组 边
                        if table_triplet and table_image_node:
                            table_edge1 = creat_image_edges(pdf_name, table_image_node, table_triplet, 5)
                        # 5.处理原图节点 与 表格文本内容 三元组 边
                        if table_text_node and table_image_node:
                            table_edge2 = creat_image_edges(pdf_name, table_image_node, table_text_node, 5)
                        # 6.表格文本内容 三元组 与 表格标题脚注 段落三元组 边
                        if table_triplet and table_text_node:
                            table_edge3 = creat_image_edges(pdf_name, table_text_node, table_triplet, 5)

                        # 7.合并表格段落的子图
                        subgraph_seg.extend(table_triplet)
                        subgraph_seg.extend(table_text_node)
                        subgraph_seg.extend(table_image_node)
                        subgraph_seg.extend(table_edge1)
                        subgraph_seg.extend(table_edge2)
                        subgraph_seg.extend(table_edge3)

                        text_list = merged_text + ([table_body] if table_body else [])
                        if table_image_description:
                            text_list.append(table_image_description)




                    # 创建本段的锚节点，与本段以及上一段的锚节点建立 anchor_edge
                    anchor_node = create_anchor_node(pdf_name=pdf_name, segment_text=text_list, 
                                                     page_idx=page_idx, segment_idx=segment_idx, chatLLM=chatLLM)
                    anchor_edge = create_anchor_edge(pdf_name=pdf_name, 
                                                     source_triplet=subgraph_seg, 
                                                     target_triplet=anchor_node, 
                                                     local=True)
                    if anchor_node_last:
                        anchor_edge_last = create_anchor_edge(pdf_name=pdf_name,
                                                              source_triplet=anchor_node, 
                                                              target_triplet=anchor_node_last, 
                                                              local=False)
                        subgraph_page.extend(subgraph_seg)
                        subgraph_page.extend(anchor_node)
                        subgraph_page.extend(anchor_edge)
                        subgraph_page.extend(anchor_edge_last)
                    else:
                        subgraph_page.extend(subgraph_seg)
                        subgraph_page.extend(anchor_node)
                        subgraph_page.extend(anchor_edge)
                    segment_idx += 1
                else:
                    print(f"未知类型: {item_type}")
                    segment_idx += 1
                
                anchor_node_last = anchor_node      # 缓存本次段落锚节点
            anchor_node_last = anchor_node      # 跨页时缓存段落锚节点

            # 页面级子图 归档到 文档级图 graph
            graph.extend(subgraph_page)

            # 每完成一页，保存一次文档级图，防止丢失
            script_dir = Path(__file__).parent.absolute()   # 获取当前脚本所在目录
            save_path_doc = script_dir / "json_llm_extract_result_doc.json"
            save_path_doc.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
            with open(save_path_doc, "w", encoding="utf-8") as f:
                    json.dump(graph, f, ensure_ascii=False, indent=2)

        logging.info(f"原始图谱构建完成，共 {len(graph)} 个元素")
    
    else:
        graph = []
        print("Error: 路径不存在或没有可加载的文档")
        logging.error("路径不存在或没有可加载的文档")

    return graph
