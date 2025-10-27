import re
import logging
import hashlib
from typing import List, Optional
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

DEFAULT_COLLECTION = "mrag_test"

class MilvusHandler:
    """
    Milvus handler:
    - 单集合：默认名可配置（默认'mrag_test'）
    - 字段：VID(VARCHAR pk), vector(FLOAT_VECTOR, dim)
    - 支持按 partition (例如 type 名) 创建分区；如果插入时没有 partition，则写入 default partition

    变更：
    - 对所有传入的 VID 进行统一的 _normalize_vid 规范化，保证与 Nebula 的 vid 规范化策略一致（可读性 + 唯一性）
      - 清洗空白、换行、引号，连续空白替换为下划线
      - 若长度过长（>128），使用 sha256 摘要并取前 128 字符
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION, host: str = "127.0.0.1", port: str = "19530"):
        connections.connect(alias="default", host=host, port=port)
        self.collection: Optional[Collection] = None
        self._dim: Optional[int] = None
        self.collection_name = collection_name

    def _normalize_vid(self, vid: str) -> str:
        """标准化 VID：不改变原始 name 的语义（不返回 name 的变体作为外部使用），仅用于作为 Milvus 主键值。

        规则：
        - 转为字符串、去首尾空白、替换换行和回车
        - 移除单/双引号
        - 将任意连续空白替换为单个下划线（便于可读）
        - 将其他非常规字符也替换为下划线（保持与 partition 名清洗一致性）
        - 若长度 > 128，则返回 sha256(vid).hexdigest()[:128]
        """
        if not isinstance(vid, str):
            vid = str(vid)
        # vid = vid.strip().replace('\n', ' ').replace('\r', ' ')
        # vid = vid.replace('"', '').replace("'", '')
        # # 把连续空白改成单下划线
        # vid = '_'.join(vid.split())
        # vid = vid.replace(' ', '_')

        if len(vid) > 128:
            h = hashlib.sha256(vid.encode('utf-8')).hexdigest()
            vid = h[:128]
        return vid

    def create_collection(self, name: Optional[str] = None, dim: int = 1024):
        if name:
            self.collection_name = name
        self._dim = dim
        if utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name)
            print(f"[Milvus] Using existing collection: {self.collection_name}")
            return

        fields = [
            FieldSchema(name="VID", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="Entity vectors; primary key VID")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"[Milvus] Created collection: {self.collection_name} (dim={dim})")

    def reset_collection(self, embedding_dim: int = 1024):
        """显式清空并重新建 collection"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"[Milvus] Dropped existing collection: {self.collection_name}")

        fields = [
            FieldSchema(name="VID", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        ]
        schema = CollectionSchema(fields, description="Entity vectors; primary key VID")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"[Milvus] Reset collection: {self.collection_name} (dim={embedding_dim})")

    def _ensure_partition(self, partition_name: str):
        if partition_name and not self.collection.has_partition(partition_name):
            self.collection.create_partition(partition_name)
            print(f"[Milvus] Created partition: {partition_name}")

    def create_vector_index(self):
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        print("[Milvus] Index created on field 'vector'")

    def load_partition(self, partition_name: str) -> bool:
        """
        将指定的分区加载到内存中。
        """
        assert self.collection is not None, "Collection is not initialized."
        try:
            # 清理分区名，确保与插入时一致
            cleaned_name = re.sub(r'[^0-9a-zA-Z_]', '_', partition_name)
            if not self.collection.has_partition(cleaned_name):
                logging.info(f"[Milvus] Partition '{cleaned_name}' does not exist. Cannot load.")
                return False

            logging.info(f"[Milvus] Loading partition '{cleaned_name}' into memory...")
            self.collection.load(partition_names=[cleaned_name])
            logging.info(f"[Milvus] Partition '{cleaned_name}' loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"[Milvus] Error loading partition '{partition_name}': {e}")
            return False

    def release_partition(self, partition_name: str):
        """
        从内存中释放指定的分区。
        """
        assert self.collection is not None, "Collection is not initialized."
        try:
            # 清理分区名
            cleaned_name = re.sub(r'[^0-9a-zA-Z_]', '_', partition_name)
            logging.info(f"[Milvus] Releasing partition '{cleaned_name}' from memory...")
            self.collection.release(partition_names=[cleaned_name])
            logging.info(f"[Milvus] Partition '{cleaned_name}' released successfully.")
        except Exception as e:
            logging.error(f"[Milvus] Error releasing partition '{partition_name}': {e}")

    def count(self) -> int:
        return int(self.collection.num_entities)

    def search(self, vector: List[float], top_k: int = 3, partition_names: Optional[List[str]] = None) -> List[str]:
        cleaned_partition_names = None
        if partition_names:
            # 对列表中的每一个名字都执行净化操作
            cleaned_partition_names = [re.sub(r'[^0-9a-zA-Z_]', '_', name) for name in partition_names]
        #self.collection.load(partition_names=cleaned_partition_names)
        print(f"[Milvus] Searching (dim={len(vector)}) top_k={top_k} partitions={cleaned_partition_names or 'ALL'}")
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["VID"],
            partition_names=cleaned_partition_names
        )
        # results[0] 为第一个查询向量的 hits
        hits = [hit.entity.get("VID") for hit in results[0]]
        print(f"[Milvus] Search results: {hits}")
        return hits

    def insert_vectors_batch(self, VID_vector_pairs: List[tuple], partition: Optional[str] = None):
        """
        批量插入多个向量和对应的 VID
        :param items: JSON 列表，形如 [{"VID": "...", "vector": [...], "type": "..."}]
        :param partition: 可选分区名；如果为 None，则优先取 item["partition"] 或 type
        """
        VID_list, vector_list = [], []

        # 清理分区名
        if partition:
            partition = re.sub(r'[^0-9a-zA-Z_]', '_', partition)

        # 拆分批量
        VID_list = [pair[0] for pair in VID_vector_pairs]
        vector_list = [pair[1] for pair in VID_vector_pairs]

        assert self.collection is not None, "Collection not created"

        # 检查维度一致性
        if self._dim is None:
            self._dim = len(vector_list[0])
        for v in vector_list:
            if len(v) != self._dim:
                raise ValueError(f"Vector dim mismatch: expected {self._dim}, got {len(v)}")
        # 对所有 VID 做规范化，保证与单条 insert 一致
        normalized_vids = [self._normalize_vid(v) for v in VID_list]
        # 确保分区存在
        if partition:
            self._ensure_partition(partition)

        # 插入数据
        entities = [normalized_vids, vector_list]
        self.collection.insert(entities, partition_name=partition if partition else None)
        self.collection.flush()
        print(f"[Milvus] Batch inserted {len(normalized_vids)} vectors into partition={partition or 'default'}; "
            f"num_entities={self.collection.num_entities}")
