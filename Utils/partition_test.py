""" 和 Milvus 建立连接，然后针对集合（Collection）和分区（Partition）做查询、卸载、统计等管理操作。 """
import re
from pymilvus import connections, Collection, utility
from pymilvus.exceptions import MilvusException

# --- 配置 ---
COLLECTION_NAME = "llamaindex_rag2"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"


def load_partition_and_query():
  connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
  # 假设 collection_name 是你要查询的集合名，partition_name 是你要查询的分区名
  collection = Collection(COLLECTION_NAME)
  # 加载特定的分区
  partition_name = "2305.14160v4"  # 你的分区名
  if partition_name:
    # 对列表中的每一个名字都执行净化操作
    cleaned_partition_name = re.sub(r'[^0-9a-zA-Z_]', '_', partition_name)
  collection.load(partition_names=[cleaned_partition_name])

  # 查询分区中的所有节点（假设你需要获取该分区中的所有向量）
  # 你可以根据你的需求调整检索逻辑，这里只是简单的查询返回前几个节点
  query_result = collection.query(
      expr="VID like '%'",
      fields=["VID", "vector"],  # 你可以根据需要查询特定的字段
      partition_names=[cleaned_partition_name],
      limit=100  # 限制查询结果的数量，可以根据需要调整
  )
  # 输出查询结果
  for item in query_result:
      print(f"VID: {item['VID']}")

def unload_partition():
  try:
      print(f"正在连接到 Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
      connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
      print("连接成功。")

      if utility.has_collection(COLLECTION_NAME):
          print(f"找到集合: '{COLLECTION_NAME}'")
          collection = Collection(COLLECTION_NAME)
          
          print("正在从内存中释放该集合的所有分区...")
          collection.release()
          print(f"成功释放集合 '{COLLECTION_NAME}'。现在它的所有分区都已从内存卸载。")
          
      else:
          print(f"未找到集合: '{COLLECTION_NAME}'，无需操作。")

  except Exception as e:
      print(f"发生错误: {e}")

  finally:
      connections.disconnect("default")
      print("断开连接。")

def check_and_print_milvus_partitions(collection_name: str, host: str, port: str):
    """
    连接到 Milvus，查询指定集合的分区信息，并直接打印包含总数在内的结果。

    参数:
    collection_name (str): 目标集合的名称。
    host (str): Milvus 服务器的主机地址。
    port (str): Milvus 服务器的端口。
    """
    alias = f"counter_{collection_name}"  # 创建一个临时的连接别名
    
    try:
        print(f"正在连接到 Milvus ({host}:{port})...")
        connections.connect(alias=alias, host=host, port=port)
        print("连接成功。")

        if not utility.has_collection(collection_name, using=alias):
            print(f"\n[错误] 集合 '{collection_name}' 不存在。")
            return

        collection = Collection(name=collection_name, using=alias)
        partitions = collection.partitions
        partition_names = [p.name for p in partitions]
        
        # --- 打印结果 ---
        print("\n---------- 查询结果 ----------")
        
        # <--- 这里就是统计和打印分区数量的地方
        partition_count = len(partition_names)
        
        
        if partition_names:
            print("分区列表:")
            for name in partition_names:
                print(f"  - {name}")
        else:
            # 如果列表为空，上面已经打印了 "找到 0 个分区"，这里可以加一句提示
            print("该集合中没有用户创建的分区。")
        print("----------------------------")
        print(f"✅ 在集合 '{collection_name}' 中总共找到 {partition_count} 个分区。")  

    except MilvusException as e:
        print(f"[Milvus 异常] 查询分区时发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        if connections.has_connection(alias):
            print("\n正在断开与 Milvus 的连接。")
            connections.disconnect(alias)

def drop_collection(collection_name: str, host: str, port: str):
    """
    彻底删除指定的 Milvus 集合（不可恢复！）
    - 会删除该集合及其所有分区与数据
    """
    alias = f"drop_{collection_name}"
    try:
        print(f"⚠️ 重要：即将删除集合 '{collection_name}'（不可恢复）。")
        print(f"正在连接到 Milvus ({host}:{port})...")
        connections.connect(alias=alias, host=host, port=port)
        print("连接成功。")

        if not utility.has_collection(collection_name, using=alias):
            print(f"[信息] 集合 '{collection_name}' 不存在，跳过删除。")
            return

        # 先释放内存（可选，但更干净）
        coll = Collection(name=collection_name, using=alias)
        try:
            coll.release()
            print(f"[信息] 已释放集合 '{collection_name}' 的内存加载。")
        except Exception as e:
            print(f"[警告] 释放集合时出现问题（可忽略继续删除）：{e}")

        # 正式删除
        utility.drop_collection(collection_name, using=alias)
        print(f"✅ 已删除集合 '{collection_name}'。")

    except MilvusException as e:
        print(f"[Milvus 异常] 删除集合时发生错误: {e}")
    except Exception as e:
        print(f"[未知错误] {e}")
    finally:
        if connections.has_connection(alias):
            connections.disconnect(alias)
            print("已断开连接。")


def get_entity_count(collection_name: str, host: str, port: str):
    """
    连接到 Milvus，查询并打印指定集合中的实体（向量）总数。

    参数:
    collection_name (str): 目标集合的名称。
    host (str): Milvus 服务器的主机地址。
    port (str): Milvus 服务器的端口。
    """
    alias = f"entity_counter_{collection_name}"  # 为此操作创建一个唯一的连接别名
    try:
        print(f"正在连接到 Milvus ({host}:{port})...")
        connections.connect(alias=alias, host=host, port=port)
        print("连接成功。")

        if not utility.has_collection(collection_name, using=alias):
            print(f"\n[错误] 集合 '{collection_name}' 不存在，无法查询。")
            return

        collection = Collection(name=collection_name, using=alias)
        
        # 在查询前，最好刷新（flush）集合以确保所有插入的数据都已落盘
        # 这能保证获取到最准确的实体数量
        print("正在刷新集合以确保数据一致性...")
        collection.flush()
        print("刷新完成。")

        num_entities = collection.num_entities
        print("\n---------- 查询结果 ----------")
        print(f"✅ 集合 '{collection_name}' 中共有 {num_entities} 个实体。")
        print("----------------------------")

    except MilvusException as e:
        print(f"[Milvus 异常] 查询实体数量时发生错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    finally:
        if connections.has_connection(alias):
            print("\n正在断开与 Milvus 的连接。")
            connections.disconnect(alias)

if __name__ == "__main__":
    load_partition_and_query()
    #unload_partition()
    #drop_collection(COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT)  #删除整个集合
    # check_and_print_milvus_partitions(
    #     collection_name=COLLECTION_NAME,
    #     host=MILVUS_HOST,
    #     port=MILVUS_PORT
    # )

    # get_entity_count(
    #      collection_name=COLLECTION_NAME,
    #      host=MILVUS_HOST,
    #      port=MILVUS_PORT
    # )