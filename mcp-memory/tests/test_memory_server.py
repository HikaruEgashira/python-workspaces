import pytest
from mcp_memory import MemoryServer, Entity, Relation, Observation
import os
import tempfile

@pytest.fixture
def memory_file():
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    os.unlink(path)

@pytest.fixture
def server(memory_file):
    return MemoryServer(memory_file)

def test_create_entities(server):
    # 基本的なエンティティ作成
    entities = [
        Entity(name="John", entity_type="person", observations=["Speaks English"]),
        Entity(name="Acme", entity_type="organization")
    ]
    server.create_entities(entities)
    
    graph = server.read_graph()
    assert len(graph["entities"]) == 2
    assert "John" in graph["entities"]
    assert "Acme" in graph["entities"]
    assert graph["entities"]["John"].observations == ["Speaks English"]

    # 重複するエンティティの作成（無視されることを確認）
    duplicate_entity = [Entity(name="John", entity_type="company", observations=["Different type"])]
    server.create_entities(duplicate_entity)
    
    graph = server.read_graph()
    assert graph["entities"]["John"].entity_type == "person"
    assert graph["entities"]["John"].observations == ["Speaks English"]

def test_create_relations(server):
    entities = [
        Entity(name="John", entity_type="person"),
        Entity(name="Acme", entity_type="organization")
    ]
    server.create_entities(entities)
    
    relations = [
        Relation(from_entity="John", to_entity="Acme", relation_type="works_at")
    ]
    server.create_relations(relations)
    
    graph = server.read_graph()
    assert len(graph["relations"]) == 1
    assert graph["relations"][0].from_entity == "John"
    assert graph["relations"][0].to_entity == "Acme"

def test_add_observations(server):
    # 基本的な観察の追加
    entity = Entity(name="John", entity_type="person", observations=["Speaks English"])
    server.create_entities([entity])
    
    observations = [
        Observation(entity_name="John", contents=["Lives in Tokyo", "Speaks English"])
    ]
    added = server.add_observations(observations)
    
    assert "John" in added
    assert len(added["John"]) == 1
    assert "Lives in Tokyo" in added["John"]
    assert len(server.entities["John"].observations) == 2

    # 存在しないエンティティへの観察の追加（無視されることを確認）
    invalid_observations = [
        Observation(entity_name="NonExistent", contents=["Should not be added"])
    ]
    added = server.add_observations(invalid_observations)
    assert "NonExistent" not in added

    # 重複する観察の追加（無視されることを確認）
    duplicate_observations = [
        Observation(entity_name="John", contents=["Lives in Tokyo"])
    ]
    added = server.add_observations(duplicate_observations)
    assert "John" in added
    assert len(added["John"]) == 0

def test_delete_entities(server):
    # テストデータのセットアップ
    entities = [
        Entity(name="John", entity_type="person", observations=["Speaks English"]),
        Entity(name="Acme", entity_type="organization"),
        Entity(name="Jane", entity_type="person")
    ]
    server.create_entities(entities)
    
    relations = [
        Relation(from_entity="John", to_entity="Acme", relation_type="works_at"),
        Relation(from_entity="Jane", to_entity="Acme", relation_type="works_at")
    ]
    server.create_relations(relations)
    
    # エンティティの削除とカスケード削除の確認
    server.delete_entities(["John"])
    graph = server.read_graph()
    
    assert "John" not in graph["entities"]
    assert len([r for r in graph["relations"] if r.from_entity == "John" or r.to_entity == "John"]) == 0
    assert len([r for r in graph["relations"] if r.from_entity == "Jane"]) == 1

    # 存在しないエンティティの削除（エラーが発生しないことを確認）
    server.delete_entities(["NonExistent"])
    assert len(graph["entities"]) == 2

def test_search_nodes(server):
    # テストデータのセットアップ
    entities = [
        Entity(name="John Smith", entity_type="person", observations=["Speaks English"]),
        Entity(name="Jane Doe", entity_type="person", observations=["Speaks Japanese"]),
        Entity(name="Japanese Restaurant", entity_type="business", observations=["Opens at 9AM"])
    ]
    server.create_entities(entities)
    
    # 観察内容での検索
    results = server.search_nodes("Japanese")
    assert len(results["entities"]) == 2
    assert "Jane Doe" in results["entities"]
    assert "Japanese Restaurant" in results["entities"]

    # エンティティ名での検索
    results = server.search_nodes("Smith")
    assert len(results["entities"]) == 1
    assert "John Smith" in results["entities"]

    # エンティティタイプでの検索
    results = server.search_nodes("business")
    assert len(results["entities"]) == 1
    assert "Japanese Restaurant" in results["entities"]

    # 存在しない検索語での検索
    results = server.search_nodes("NonExistent")
    assert len(results["entities"]) == 0

def test_delete_observations(server):
    # テストデータのセットアップ
    entity = Entity(name="John", entity_type="person", observations=["Speaks English", "Lives in Tokyo", "Likes coffee"])
    server.create_entities([entity])

    # 特定の観察の削除
    deletions = [Observation(entity_name="John", contents=["Lives in Tokyo"])]
    server.delete_observations(deletions)
    
    graph = server.read_graph()
    assert len(graph["entities"]["John"].observations) == 2
    assert "Lives in Tokyo" not in graph["entities"]["John"].observations
    assert "Speaks English" in graph["entities"]["John"].observations
    assert "Likes coffee" in graph["entities"]["John"].observations

    # 存在しないエンティティの観察の削除（エラーが発生しないことを確認）
    invalid_deletions = [Observation(entity_name="NonExistent", contents=["Should not error"])]
    server.delete_observations(invalid_deletions)

    # 存在しない観察の削除（エラーが発生しないことを確認）
    non_existent_obs = [Observation(entity_name="John", contents=["NonExistent Observation"])]
    server.delete_observations(non_existent_obs)
    assert len(graph["entities"]["John"].observations) == 2

def test_delete_relations(server):
    # テストデータのセットアップ
    entities = [
        Entity(name="John", entity_type="person"),
        Entity(name="Jane", entity_type="person"),
        Entity(name="Acme", entity_type="organization")
    ]
    server.create_entities(entities)
    
    relations = [
        Relation(from_entity="John", to_entity="Jane", relation_type="knows"),
        Relation(from_entity="John", to_entity="Acme", relation_type="works_at"),
        Relation(from_entity="Jane", to_entity="Acme", relation_type="works_at")
    ]
    server.create_relations(relations)

    # 特定のリレーションの削除
    to_delete = [Relation(from_entity="John", to_entity="Jane", relation_type="knows")]
    server.delete_relations(to_delete)
    
    graph = server.read_graph()
    assert len(graph["relations"]) == 2
    assert all(r.relation_type != "knows" for r in graph["relations"])

    # 存在しないリレーションの削除（エラーが発生しないことを確認）
    non_existent = [Relation(from_entity="John", to_entity="NonExistent", relation_type="invalid")]
    server.delete_relations(non_existent)
    assert len(graph["relations"]) == 2

def test_open_nodes(server):
    # テストデータのセットアップ
    entities = [
        Entity(name="John", entity_type="person"),
        Entity(name="Jane", entity_type="person"),
        Entity(name="Acme", entity_type="organization")
    ]
    server.create_entities(entities)
    
    relations = [
        Relation(from_entity="John", to_entity="Jane", relation_type="knows"),
        Relation(from_entity="John", to_entity="Acme", relation_type="works_at"),
        Relation(from_entity="Jane", to_entity="Acme", relation_type="works_at")
    ]
    server.create_relations(relations)
    
    # 特定のノードとその関係の取得
    results = server.open_nodes(["John", "Jane"])
    assert len(results["entities"]) == 2
    assert len(results["relations"]) == 1
    assert results["relations"][0].relation_type == "knows"

    # 存在しないノードの取得（スキップされることを確認）
    results = server.open_nodes(["John", "NonExistent"])
    assert len(results["entities"]) == 1
    assert "John" in results["entities"]

    # 空のリストでの取得
    results = server.open_nodes([])
    assert len(results["entities"]) == 0
    assert len(results["relations"]) == 0
