from typing import Dict, List, Tuple
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import ttypes
import time
import hashlib


# 预定义 TAG 名称（以实际在数据库中需要的名字为准）
PREDEFINED_TAGS = [
    "ENTITY_NODE",
    "SEGMENT ANCHOR NODE",
    "ORIGINAL_IMAGE",
    "EQUATION_NODE",
    "TABLE_NODE",
]

# 映射外部类型到预定义 TAG（保证映射结果与 PREDEFINED_TAGS 中的名字一致）
TYPE_TO_TAG_ALIASES = {
    "TABLE NODE": "TABLE_NODE",
    "TABLE_NODE": "TABLE_NODE",
    "ORIGINAL IMAGE": "ORIGINAL_IMAGE",
    "ORIGINAL_IMAGE": "ORIGINAL_IMAGE",
    "SEGMENT ANCHOR NODE": "SEGMENT ANCHOR NODE",
    "SEGMENT_ANCHOR_NODE": "SEGMENT ANCHOR NODE",
    "EQUATION NODE": "EQUATION_NODE",
    "EQUATION_NODE": "EQUATION_NODE",
    "ENTITY NODE": "ENTITY_NODE",
    "ENTITY_NODE": "ENTITY_NODE",
}

ENTITY_TAG = PREDEFINED_TAGS[0]
SPECIAL_TAGS = set(PREDEFINED_TAGS[1:])


class NebulaHandler:
    """
    Nebula handler:
    - 固定五个预定义 TAG
    - ENTITY_NODE 顶点 vid_type=FIXED_STRING(256)，使用 docid + entityID/name 作为 vid（规范化）
    - ENTITY_NODE 附加字段：name, description, type, entityID, chunkID
    - 统一边类型：RelatesTo(relationship string, relationship_strength double)

    重要：
    - name_to_vid 的 key 使用 (ref_doc_id, original_name_string) 以保证不同文档中相同 name 映射到不同 vid
    - 在插入实体时，会同时把 (ref_doc_id, entityID) 与 (ref_doc_id, name) 两个 key 都映射到相同的 vid（如果 entityID 存在）
    """

    def __init__(self, space_name: str = "mrag_test", host: str = "127.0.0.1", port: int = 9669, user: str = "root", password: str = "nebula"):
        self.space_name = space_name
        config = Config()
        self.pool = ConnectionPool()
        ok = self.pool.init([(host, port)], config)
        if not ok:
            raise RuntimeError("Nebula ConnectionPool init failed")
        self.session = self.pool.get_session(user, password)
        self._ensure_space_exists()

        # name -> entityID 映射。Key 为 (ref_doc_id, name_or_entityID)
        # 目的：支持 JSON 中关系以 name->name 形式给出，但在数据库中需要 vid->vid
        self.name_to_vid: Dict[Tuple[str, str], str] = {}

    def _normalize_vid(self, vid: str) -> str:
        """确保 Nebula vid 符合 FIXED_STRING(256) 要求并做简单清洗。

        注意：这个函数**只改变 vid 字符串**，不改变原始的 name 字段。name 仍然保存在映射键中原样。
        """
        if not isinstance(vid, str):
            vid = str(vid)

        # # 基本清洗：去掉首尾空白, 删除换行, 用下划线替换空格, 移除引号
        # vid = vid.strip().replace('\n', ' ').replace('\r', ' ')
        # vid = vid.replace('"', '').replace("'", '')
        # # 将长的连续空白替换为单下划线，避免空格导致的不同表示
        # vid = '_'.join(vid.split())
        # # 把所有非字母数字下划线点破折号之类的字符替换为下划线，避免特殊字符引发问题
        # vid = vid.replace(' ', '_')

        # 如果过长，用 sha256 摘要并取前 128 字符（在 FIXED_STRING(256) 下安全）
        if len(vid) > 128:
            h = hashlib.sha256(vid.encode('utf-8')).hexdigest()
            vid = h[:128]

        return vid

    def _exec(self, nGQL: str, retries: int = 20, retry_interval: float = 1.0):
        for attempt in range(retries):
            try:
                res = self.session.execute(nGQL)
                if res.error_code() == ttypes.ErrorCode.SUCCEEDED:
                    return res
                if "Not the leader" in res.error_msg():
                    print(f"[WARN] Not the leader, 第 {attempt+1} 次重试 ...\n[NGQL]: {nGQL}")
                else:
                    print(f"[WARN] 执行失败，错误信息: {res.error_msg()}，第 {attempt+1} 次重试 ...\n[NGQL]: {nGQL}")
            except Exception as e:
                print(f"[WARN] 异常: {str(e)}，第 {attempt+1} 次重试 ...\n[NGQL]: {nGQL}")

            time.sleep(retry_interval)

        raise RuntimeError(f"[ERROR] 重试 {retries} 次后仍然失败: {nGQL}")

    def _ensure_space_exists(self):
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}(partition_num=1, replica_factor=1, vid_type=FIXED_STRING(256));'
        )
        # 等待 space 出现在 SHOW SPACES
        for i in range(30):
            try:
                res = self._exec("SHOW SPACES;")
                spaces = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
                if self.space_name in spaces:
                    break
                time.sleep(1)
            except RuntimeError:
                time.sleep(1)
                continue
        else:
            raise RuntimeError(f"Space {self.space_name} 一直未出现在 SHOW SPACES")
        self._exec(f'USE {self.space_name};')

    def create_schema(self):
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}('
            f'partition_num=1, replica_factor=1, vid_type=FIXED_STRING(256));'
        )

        for i in range(30):
            try:
                self._exec(f'USE {self.space_name};')
                print(f"[Nebula] Space {self.space_name} 已确认可用 ✅")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直无法 USE")

        # 创建 TAG & EDGE（不会覆盖已有数据）
        # 使用 PREDEFINED_TAGS 的名字
        self._exec(
            f'CREATE TAG IF NOT EXISTS `{ENTITY_TAG}`(name string, description string, type string, entityID string, chunkID string, ref_doc_id string);'
        )
        for tag in SPECIAL_TAGS:
            self._exec(
                f'CREATE TAG IF NOT EXISTS `{tag}`(name string, description string, type string, entityID string, chunkID string, ref_doc_id string);'
            )
        self._exec(
            'CREATE EDGE IF NOT EXISTS RelatesTo(relationship string, relationship_strength double, ref_doc_id string);'
        )

        # 确认 TAG 已存在
        for i in range(30):
            res = self._exec("SHOW TAGS;")
            tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            if "ENTITY_NODE" in tags:
                print("[Nebula] ENTITY_NODE schema 已确认存在 ✅")
                break
            time.sleep(1)
        else:
            raise RuntimeError("ENTITY_NODE schema 未成功创建")

    def reset_space(self):
        print(f"[DEBUG] 尝试删除空间 {self.space_name}")
        self._exec(f'DROP SPACE IF EXISTS {self.space_name};')
        time.sleep(2)

        print(f"[DEBUG] 重新创建空间 {self.space_name}")
        self._exec(
            f'CREATE SPACE IF NOT EXISTS {self.space_name}('
            f'partition_num=1, replica_factor=1, vid_type=FIXED_STRING(256));'
        )

        for i in range(30):
            res = self._exec("SHOW SPACES;")
            spaces = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            if self.space_name in spaces:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直未出现在 SHOW SPACES")

        for i in range(30):
            try:
                self._exec(f'USE {self.space_name};')
                print(f"[Nebula] Space {self.space_name} 已确认可用 ✅")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"Space {self.space_name} 一直无法 USE")

        # 创建 TAG & EDGE
        self._exec(
            f'CREATE TAG IF NOT EXISTS `{ENTITY_TAG}`(name string, description string, type string, entityID string, chunkID string, ref_doc_id string);'
        )
        for tag in SPECIAL_TAGS:
            self._exec(
                f'CREATE TAG IF NOT EXISTS `{tag}`(name string, description string, type string, entityID string, chunkID string, ref_doc_id string);'
            )
        self._exec(
            'CREATE EDGE IF NOT EXISTS RelatesTo(relationship string, relationship_strength double, ref_doc_id string);'
        )

        for i in range(30):
            res = self._exec("SHOW TAGS;")
            tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
            print("[DEBUG] 当前 TAGS:", tags)
            if "ENTITY_NODE" in tags:
                print("[Nebula] ENTITY_NODE schema 已确认创建成功 ✅")
                break
            time.sleep(1)
        else:
            raise RuntimeError("ENTITY_NODE schema 未成功创建")

        for tag in [ENTITY_TAG] + list(SPECIAL_TAGS):
            for i in range(10):
                try:
                    self._exec(f'DESC TAG `{tag}`;')
                    print(f"[Nebula] DESC TAG {tag} 成功，Storage 已同步 ✅")
                    break
                except Exception as e:
                    print(f"[DEBUG] DESC TAG {tag} 第 {i+1} 次失败，等待同步: {e}")
                    time.sleep(1)
            else:
                raise RuntimeError(f"{tag} schema 一直未同步到 Storage")

        for i in range(10):
            try:
                self._exec('DESC EDGE RelatesTo;')
                print("[Nebula] DESC EDGE RelatesTo 成功，Storage 已同步 ✅")
                break
            except Exception as e:
                print(f"[DEBUG] DESC EDGE RelatesTo 第 {i+1} 次失败，等待同步: {e}")
                time.sleep(1)
        else:
            raise RuntimeError("RelatesTo edge schema 一直未同步到 Storage")

        # Dummy 节点写入测试（静默重试，直到成功）
        test_vid = self._normalize_vid("DummyTestNode001")
        last_err = None

        for i in range(60):
            try:
                nGQL = (
                    f'INSERT VERTEX `{ENTITY_TAG}`(name, description, type, entityID, chunkID, ref_doc_id) '
                    f'VALUES "{test_vid}":("tmp_name", "tmp_desc", "tmp_type", "0", "0", "0");'
                )
                self._exec(nGQL)
                self._exec(f'FETCH PROP ON `{ENTITY_TAG}` "{test_vid}" YIELD properties(vertex);')
                self._exec(f'DELETE VERTEX "{test_vid}";')
                print(f"[Nebula] Dummy 节点写入成功 ✅ (第 {i+1} 次尝试)")
                break
            except Exception as e:
                msg = str(e)
                last_err = msg
                time.sleep(1)
                continue
        else:
            print(f"[WARN] Dummy 节点写入在 60 次尝试后仍未成功，最后错误: {last_err}")

    # ---------- 辅助：根据 type_name 得到目标 tag ----------
    def _resolve_tag(self, type_name: str) -> str:
        if not type_name:
            return ENTITY_TAG
        k = type_name.strip().upper()
        # 先尝试直接在映射表中匹配
        mapped = TYPE_TO_TAG_ALIASES.get(k, None)
        if mapped and mapped in ([ENTITY_TAG] + list(SPECIAL_TAGS)):
            return mapped
        # 再尝试直接使用原始（可能用户已经给出正确的 tag 名）
        if type_name in ([ENTITY_TAG] + list(SPECIAL_TAGS)):
            return type_name
        # 兜底
        return ENTITY_TAG

    # -------------------  批量写入-------------------
    def insert_entities_bulk(self, entities: List[Dict], batch_size: int = 1000):
        """批量插入实体。会同时建立 (ref_doc_id, entityID) 与 (ref_doc_id, name) 到 VID 的映射。"""
        self._exec(f'USE {self.space_name};')

        res = self._exec("SHOW TAGS;")
        tags = [row.values[0].get_sVal().decode("utf-8") for row in res.rows()]
        print("[DEBUG] insert_entities_bulk 当前 TAGS:", tags)
        if ENTITY_TAG not in tags:
            raise RuntimeError("[ERROR] ENTITY_NODE schema 未找到，无法批量插入")

        buffer: Dict[str, List[str]] = {}

        def esc(s: str) -> str:
            if s is None:
                return ""
            return str(s).replace('"', '\\"')

        for e in entities:
            entityID = e.get("entityID", "")
            ref_doc_id = e.get("ref_doc_id", "")
            name = e.get("name", "")
            if not entityID and not name:
                print(f"[WARN] 跳过无效实体 (既无 entityID 又无 name): {e}")
                continue
            if not ref_doc_id:
                print(f"[WARN] 跳过无效实体 (缺少 ref_doc_id): {e}")
                continue

            # raw VID 来源：优先 entityID，否则用 name
            raw_vid_source = f"{ref_doc_id}_{entityID if entityID else name}"
            VID = self._normalize_vid(raw_vid_source)

            # 建立映射：(ref_doc_id, entityID) 与 (ref_doc_id, name)
            if entityID:
                self.name_to_vid[(ref_doc_id, entityID)] = VID
            if name:
                # 保证不改变原始 name 字符串作为键
                self.name_to_vid[(ref_doc_id, name)] = VID

            type_name = e.get("type", "")
            desc = e.get("description", "")
            chunkID = e.get("chunkID", "")

            target_tag = self._resolve_tag(type_name)

            values = f'"{VID}":("{esc(name)}", "{esc(desc)}", "{esc(type_name)}", "{esc(entityID)}", "{esc(chunkID)}", "{esc(ref_doc_id)}")'

            if target_tag not in buffer:
                buffer[target_tag] = []
            buffer[target_tag].append(values)

            # 分批插入
            if len(buffer[target_tag]) >= batch_size:
                self._flush_entities(buffer[target_tag], target_tag)
                buffer[target_tag].clear()

        # 插入剩余的
        for tag, values in buffer.items():
            if values:
                self._flush_entities(values, tag)

    def _flush_entities(self, values: List[str], tag: str, retries: int = 5, retry_interval: float = 1.0):
        """批量插入实体"""
        if tag == ENTITY_TAG:
            nGQL = f'INSERT VERTEX `{ENTITY_TAG}`(name, description, type, entityID, chunkID, ref_doc_id) VALUES {",".join(values)};'
        else:
            nGQL = f'INSERT VERTEX `{tag}`(name, description, type, entityID, chunkID, ref_doc_id) VALUES {",".join(values)};'

        last_err = None
        for _ in range(retries):
            try:
                self._exec(nGQL)
                print(f"[Nebula] Bulk inserted {len(values)} entities into {tag}")
                return
            except Exception as e:
                msg = str(e)
                last_err = msg
                if "Not the leader" in msg or "Storage Error" in msg:
                    time.sleep(retry_interval)
                    continue
                raise
        raise RuntimeError(f"[ERROR] 插入 {tag} 批量数据失败超过 {retries} 次，最后错误: {last_err}")

    def insert_relations_bulk(self, relations: List[Dict], batch_size: int = 1000):
        """批量插入边，relations 中 source/target 应为 name 或 entityID（以 JSON 的实际字段为准）。"""
        values: List[str] = []

        for r in relations:
            ref_doc_id = r.get("ref_doc_id", "")
            entityID_src = r.get("source", "")
            entityID_dst = r.get("target", "")

            if not ref_doc_id or not entityID_src or not entityID_dst:
                print(f"[WARN] 跳过无效关系: {r}")
                continue

            # 优先从映射中解析（使用未改变的 name 字符串），若未找到则使用默认构造的 vid
            src_vid = self.name_to_vid.get((ref_doc_id, entityID_src))
            if src_vid is None:
                src_vid = self._normalize_vid(f"{ref_doc_id}_{entityID_src}")
                print(f"[WARN] 在映射中未找到 src {entityID_src}，使用默认 vid {src_vid}")

            dst_vid = self.name_to_vid.get((ref_doc_id, entityID_dst))
            if dst_vid is None:
                dst_vid = self._normalize_vid(f"{ref_doc_id}_{entityID_dst}")
                print(f"[WARN] 在映射中未找到 dst {entityID_dst}，使用默认 vid {dst_vid}")

            rel = str(r.get("relationship", "")).replace('"', '\\"')
            strength = float(r.get("relationship_strength", 0.0))

            values.append(f'"{src_vid}"->"{dst_vid}":("{rel}", {strength}, "{ref_doc_id}")')

            if len(values) >= batch_size:
                self._flush_relations(values)
                values.clear()

        if values:
            self._flush_relations(values)

    def _flush_relations(self, values: List[str], retries: int = 5, retry_interval: float = 1.0):
        self._exec(f'USE {self.space_name};')
        nGQL = f'INSERT EDGE RelatesTo(relationship, relationship_strength, ref_doc_id) VALUES {",".join(values)};'

        last_err = None
        for _ in range(retries):
            try:
                self._exec(nGQL)
                print(f"[Nebula] Bulk inserted {len(values)} relations")
                return
            except Exception as e:
                msg = str(e)
                last_err = msg
                time.sleep(retry_interval)
                continue

        raise RuntimeError(f"[ERROR] 插入关系批量数据失败超过 {retries} 次，最后错误: {last_err}")

    # ---------- 检索邻居 ----------
    def fetch_neighbors_2_hops(self, VID: str) -> Dict[str, List[str]]:
        self._exec(f'USE {self.space_name};')

        # q1 = f'GO 1 STEPS FROM "{VID}" OVER RelatesTo BIDIRECT YIELD DISTINCT dst(edge) as dst;'
        q1 = f'(GO 1 STEPS FROM "{VID}" OVER RelatesTo YIELD dst(edge) AS neighbor) UNION (GO 1 STEPS FROM "{VID}" OVER RelatesTo REVERSELY YIELD src(edge) AS neighbor);'
        r1 = self._exec(q1)
        hop1 = {row.values[0].get_sVal().decode("utf-8") for row in r1.rows()}
        hop1.add(VID)
        hop1 = sorted(hop1)

        q2 = f'GO 2 STEPS FROM "{VID}" OVER RelatesTo BIDIRECT YIELD DISTINCT dst(edge) as dst;'
        r2 = self._exec(q2)
        hop2_raw = {row.values[0].get_sVal().decode("utf-8") for row in r2.rows()}
        hop2 = sorted([v for v in hop2_raw if v != VID and v not in hop1])

        print(f"[Nebula] Neighbors for {VID}: 1-hop={hop1}, 2-hop={hop2}")
        return {"1-hop": hop1, "2-hop": hop2}

    def fetch_neighbors_1_hop(self, VID: str) -> Dict[str, List[str]]:
        self._exec(f'USE {self.space_name};')

        q1 = f'(GO 1 STEPS FROM "{VID}" OVER RelatesTo YIELD dst(edge) AS neighbor) UNION (GO 1 STEPS FROM "{VID}" OVER RelatesTo REVERSELY YIELD src(edge) AS neighbor);'
        r1 = self._exec(q1)
        hop1 = {row.values[0].get_sVal().decode("utf-8") for row in r1.rows()}
        hop1.add(VID)
        hop1 = sorted(hop1)

        print(f"[Nebula] Neighbors for {VID}: 1-hop={hop1}")
        return {"1-hop": hop1}

    # ---------- 统计工具 ----------
    def get_node_count(self):
        query_entity_node = "MATCH (v:ENTITY_NODE) RETURN COUNT(v) AS entity_node_count"
        res_entity_node = self._exec(query_entity_node)
        entity_node_count = res_entity_node.rows()[0].values[0].get_iVal()

        query_segment_anchor_node = "MATCH (v:`SEGMENT ANCHOR NODE`) RETURN COUNT(v) AS segment_anchor_node_count"
        res_segment_anchor_node = self._exec(query_segment_anchor_node)
        segment_anchor_node_count = res_segment_anchor_node.rows()[0].values[0].get_iVal()

        query_original_image = "MATCH (v:`ORIGINAL_IMAGE`) RETURN COUNT(v) AS original_image_node_count"
        res_original_image = self._exec(query_original_image)
        original_image_node_count = res_original_image.rows()[0].values[0].get_iVal()

        return {
            "ENTITY_NODE": entity_node_count,
            "SEGMENT ANCHOR NODE": segment_anchor_node_count,
            "ORIGINAL_IMAGE": original_image_node_count
        }

    def get_edge_count(self):
        query = "MATCH ()-[e:RelatesTo]->() RETURN COUNT(e)"
        res = self._exec(query)
        return res.rows()[0].values[0].get_iVal()
    
    def close(self):
        try:
            self.session.release()
        finally:
            self.pool.close()