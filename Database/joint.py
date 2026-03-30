import json
from typing import Dict, List, Optional
# from DocRAG.Database.nebula import NebulaHandler
# from DocRAG.Database.milvus import MilvusHandler

try:
    # 当作为模块导入时
    from .nebula import NebulaHandler
    from .milvus import MilvusHandler
except ImportError:
    # insert.py
    from nebula import NebulaHandler
    from milvus import MilvusHandler
from sentence_transformers import SentenceTransformer
import torch

class JointHandler:
    """
    统一协调：
    - ingest_graph_from_json(json_path): 读取 json 文件（list of dict），写入 Nebula（name,type,description,entityID,chunkID）与关系
    - ingest_vectors_from_json(json_path): 读取 json 文件（list of dict），每个条目至少包含 entityID 和 vector，写入 Milvus
    - build_vectors_from_graph(encoder)：用已缓存的实体生成向量并写入 Milvus（若有 type，则作为 partition）
    - search_neighbors_by_vector(vector, top_k, partition=None)：在 Milvus 中检索并返回对应 Nebula 的 1/2 跳邻居
    """

    def __init__(self,
                 space_name: str = "mrag_test",
                 nebula_host: str = "127.0.0.1", nebula_port: int = 9669, nebula_user: str = "root", nebula_password: str = "nebula",
                 milvus_host: str = "127.0.0.1", milvus_port: str = "19530",
                 collection_name: str = "mrag_test"):
        self.nebula = NebulaHandler(space_name=space_name, host=nebula_host, port=nebula_port, user=nebula_user, password=nebula_password)
        self.milvus = MilvusHandler(collection_name=collection_name, host=milvus_host, port=milvus_port)
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        # 缓存：entityID -> {type, name, description, chunkID}
        self._entity_cache: Dict[str, Dict] = {}

    def setup(self, embedding_dim: int = 1024):
        """
        初始化数据库环境（仅保证 schema 存在，不清空数据）
        """
        # Nebula 只保证 schema 存在
        self.nebula.create_schema()

        # Milvus collection 初始化
        self.milvus.create_collection(name=self.milvus.collection_name, dim=embedding_dim)



    # ---------- 图数据接口（从 json 文件读取） ----------
    def ingest_graph_from_json(self, json_path: str, embedding_dim: int = 1024, bulk: bool = True):
        """
        期望 json 文件内容为 List[Dict]，实体条目与关系条目混合：
        实体条目样例：{"entityID": "E1", "type": "Article", "name": "Title", "description": "desc", "chunkID": "C1"}
        关系条目样例：{"relationship": "cites", "source": "E1", "target": "E2", "relationship_strength": 0.9}
        """
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        #self.ingest_graph_data(items)
        if bulk:  ###  新增：增加 bulk 参数，调用批量插入
            self.ingest_graph_data_bulk(items)
        else:     ###  新增：兼容原始逐条方式
            self.ingest_graph_data(items)

    def ingest_graph_data(self, items: List[Dict]):
        self.nebula.reset_space()
        for obj in items:
            # 实体检测（以 entityID + type + name 为实体）
            if all(k in obj for k in ("entityID", "type", "name")):
                entityID = str(obj["entityID"])
                type_name = str(obj["type"])
                name = str(obj.get("name", ""))
                desc = str(obj.get("description", ""))
                chunkID = str(obj.get("chunkID", ""))
                ref_doc_id = str(obj.get("ref_doc_id", "DOC"))  
                VID = f"{ref_doc_id}_{entityID}"
                self.nebula.insert_entity(entityID=entityID, type_name=type_name, name=name, ref_doc_id=ref_doc_id, description=desc, chunkID=chunkID)
                # 缓存以便后续生成向量
                self._entity_cache[VID] = {"type": type_name, "name": name, "description": desc, "entityID": entityID, "chunkID": chunkID, "ref_doc_id": ref_doc_id}
            # 关系检测（保留原始关系键名）
            elif all(k in obj for k in ("relationship", "source", "target", "relationship_strength")):
                src = str(obj["source"])
                dst = str(obj["target"])
                ref_doc_id = str(obj.get("ref_doc_id", "DOC"))
                rel = str(obj.get("relationship", ""))
                strength = float(obj.get("relationship_strength", 0.0))
                self.nebula.insert_relation(entityID_src=src, entityID_dst=dst, relationship=rel, relationship_strength=strength, ref_doc_id=ref_doc_id)
            else:
                print(f"[Joint] WARN: Unrecognized graph item schema: {obj}")

    def ingest_graph_data_bulk(self, items: List[Dict], entity_batch: int = 1000, relation_batch: int = 2000, reset: bool = False):
        """
        批量插入（bulk 模式）
        - reset=True: 清空 Nebula 空间后再导入（危险，数据会丢）
        - reset=False: 安全模式，仅确保 schema 存在
        """
        if reset:
            self.nebula.reset_space()
        else:
            self.nebula.create_schema()

        entities, relations = [], []

        for obj in items:
            if all(k in obj for k in ("entityID", "type", "name")):
                entityID = str(obj["entityID"])
                ref_doc_id = str(obj.get("ref_doc_id", "DOC"))
                VID = f"{ref_doc_id}_{entityID}"
                obj["VID"] = VID

                entities.append(obj)
                self._entity_cache[VID] = {
                    "type": obj["type"], "name": obj["name"],
                    "description": obj.get("description", ""),"entityID": obj.get("entityID", ""), 
                    "chunkID": obj.get("chunkID", ""),"ref_doc_id": obj.get("ref_doc_id", ""), "VID": VID,
                }
            elif all(k in obj for k in ("relationship", "source", "target", "relationship_strength")):
                relations.append(obj)
            else:
                print(f"[Joint] WARN: Unrecognized graph item schema: {obj}")

        if entities:
            self.nebula.insert_entities_bulk(entities, batch_size=entity_batch)
        if relations:
            self.nebula.insert_relations_bulk(relations, batch_size=relation_batch)

    # ---------- 向量数据接口（从 json 文件读取） ----------
    def ingest_vectors_from_json(self, json_path: str, embedding_dim: int = 1024):
        """
        期望 json 文件内容为 List[Dict]，每个条目至少包含 {"entityID": "...", "vector": [...]}
        可以包含可选字段 "partition"（例如 type），若提供则在该分区写入
        """
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.ingest_vector_data(items, embedding_dim, reset=True)

    def ingest_vector_data(self, items: List[Dict], embedding_dim: int = 1024, reset: bool = True):
        """
        items：[{VID, vector, partition?, type?}]
        仅存储 VID 与 vector 到 Milvus（partition 可选）
        - 如果 type == "SEGMENT ANCHOR NODE"，也存入数据库，其他的跳过
        """
        """在导入向量数据前清空 Milvus"""
        if reset:
            self.milvus.reset_collection(embedding_dim)
        for obj in items:
            t = str(obj.get("type", ""))  # type 可选
            ref_doc_id = str(obj.get("ref_doc_id", ""))
            entityID = str(obj.get("entityID", ""))
            VID = f'{ref_doc_id}_{entityID}'

            # 只保留 ENTITY_NODE 和 SEGMENT ANCHOR NODE
            if t in ("ORIGINAL_IMAGE", "EQUATION_NODE", "TABLE_NODE"):
                continue
            
            if not t and "relationship" in obj: #排除边
                continue

            if "vector" not in obj:
                print(f"[Joint] WARN: Vector item missing required keys: {obj}")
                continue

            vector = obj["vector"]
            partition = ref_doc_id  
            self.milvus.insert_vector(VID=VID, vector=vector, partition_name=partition)

        # 创建索引（一次即可）
        self.milvus.create_vector_index()


    def ingest_vector_data_bulk(self, items: List[Dict], partition_name: Optional[str] = None, embedding_dim: int = 1024, reset: bool = False):
        """
        items：[{entityID, vector, partition?, type?}]
        仅存储 VID 与 vector 到 Milvus（partition 可选）
        - 如果 type == "SEGMENT ANCHOR NODE"，则跳过（不报 warning、不写入）
        """
        """在导入向量数据前清空 Milvus"""
        if reset:
            self.milvus.reset_collection(embedding_dim)

        VID_vector_pairs: List[tuple] = []
        partition = None
        for obj in items:
            t = str(obj.get("type", ""))  # type 可选
            ref_doc_id = str(obj.get("ref_doc_id", ""))
            entityID = str(obj.get("entityID", ""))
            VID = f'{ref_doc_id}_{entityID}'

            if ref_doc_id and partition is None:  # 保证ref_doc_id再进行赋值
                partition = ref_doc_id
            
            if t in ( "ORIGINAL_IMAGE", "EQUATION_NODE", "TABLE_NODE"):
                continue

            if not t and "relationship" in obj: #排除边
                continue

            if "vector" not in obj:
                print(f"[Joint] WARN: Vector item missing required keys: {obj}")
                continue
            vector = obj["vector"]
            VID_vector_pairs.append((VID, vector))
        
        #partition = partition_name #obj.get("partition") or t  # 如果传入 partition，则用 partition，否则用 type
        if VID_vector_pairs:
            # 关键修改：一次性批量插入，并统一使用调用方给的 partition_name（比如文件名）
            self.milvus.insert_vectors_batch(VID_vector_pairs, partition=partition)

        # 创建索引（一次即可）
        self.milvus.create_vector_index()


    # ---------- 从已缓存的图实体构建向量并写入 Milvus（示例） ----------
    def build_vectors_from_graph(self, encoder, batch_size: int = 64):
        """
        使用 encoder 对缓存的实体（name + description）编码并写入 Milvus。
        以 ref_doc_id 为 partition 的分区。
        """
        # 1) 只保留需要写入 Milvus 的节点类型
        NOT_allowed_types = {"ORIGINAL_IMAGE", "EQUATION_NODE", "TABLE_NODE"}
        ids = [eid for eid, meta in self._entity_cache.items()
           if meta.get("type") not in NOT_allowed_types]
        if not ids:
            print("[Joint] No cached entities to build vectors for.")
            return
        # 2) 按批量编码与插入
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            texts = [
                f"{self._entity_cache[eid].get('name', '')} {self._entity_cache[eid].get('description', '')}"
                for eid in batch_ids
            ]
        #texts = [self._entity_cache[e]["name"] + " " + self._entity_cache[e]["description"] for e in ids]
            vecs = encoder.encode(texts)
            for eid, v in zip(batch_ids, vecs):
                partition = self._entity_cache[eid].get("ref_doc_id")
                if hasattr(v, "tolist"):
                    v = v.tolist()
                self.milvus.insert_vector(VID=eid, vector=v, partition_name=partition)
        # 建索引
        self.milvus.create_vector_index()

    # ---------- 检索 ----------
    def search_neighbors_by_vector(self, 
                               vector,
                               top_k: int = 30,
                               top_m: int = 5,
                               top_n: int = 5,
                               partition: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        在 Milvus 中检索返回最相近的 VIDs，
        分别取前 top_m 个包含 'anchor' 的 VID 和前 top_n 个不包含 'anchor' 的 VID，
        然后对这些 VID 调用 Nebula 的邻居检索。
        
        partition: 如果传入字符串则只检索该分区（例如 type 名）；None 表示全分区
        返回：{ VID: {"1-hop":[...], "2-hop":[...]} }
        """
        part_names = [partition] if partition else None
        VIDs = self.milvus.search(vector, top_k=top_k, partition_names=part_names)

        # 按是否包含 "anchor" 分组
        anchor_vids = [vid for vid in VIDs if "anchor" in vid]
        non_anchor_vids = [vid for vid in VIDs if "anchor" not in vid]

        # 取前 top_m 和 top_n
        selected_vids = anchor_vids[:top_m] + non_anchor_vids[:top_n]

        out = {}
        for vid in selected_vids:
            # out[vid] = self.nebula.fetch_neighbors_1_hop(vid) # 直接指定1hop在后面retriever的chunk部分会报错
            out[vid] = self.nebula.fetch_neighbors_2_hops(vid)

        return out


    def search_neighbours_by_name(self,
                                 question: str,
                                 top_k: int = 3,
                                 partition: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        根据问题文本检索最相近的实体及其邻居
        
        Args:
            question: 问题文本
            top_k: 返回的最相似实体数量
            partition: 如果传入字符串则只检索该分区；None 表示全分区
            
        Returns:
            { VID: {"1-hop":[...], "2-hop":[...]} }
        """
        # 对问题进行编码
        with torch.no_grad():
            # 将问题编码为向量
            question_embedding = self.model.encode(question)
            # 转换为列表格式（如果模型输出的是 numpy 数组或 tensor）
            if hasattr(question_embedding, 'tolist'):
                question_embedding = question_embedding.tolist()
        
        # 调用向量搜索函数
        return self.search_neighbors_by_vector(
            vector=question_embedding,
            top_k=top_k,
            partition=partition
        )

    def close(self):
        self.nebula.close()
