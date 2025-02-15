import pytest
from fastapi.testclient import TestClient
from mcp_memory.api import app
from mcp_memory.models import Entity, Relation, Observation
from mcp_memory.memory_server import MemoryServer

@pytest.fixture(autouse=True)
def setup_memory_file(tmp_path):
    memory_file = tmp_path / "test_memory.json"
    app.state.server = MemoryServer(str(memory_file))
    yield memory_file

client = TestClient(app)

def test_create_entities():
    entities = [
        Entity(name="John", entity_type="person", observations=["Speaks English"]),
        Entity(name="Acme", entity_type="organization")
    ]
    response = client.post("/entities", json=[entity.model_dump() for entity in entities])
    assert response.status_code == 201
    assert response.json() == {"message": "Entities created successfully"}

def test_create_relations():
    # まずエンティティを作成
    entities = [
        Entity(name="John", entity_type="person"),
        Entity(name="Acme", entity_type="organization")
    ]
    client.post("/entities", json=[entity.model_dump() for entity in entities])

    # リレーションを作成
    relations = [
        Relation(from_entity="John", to_entity="Acme", relation_type="works_at")
    ]
    response = client.post("/relations", json=[relation.model_dump() for relation in relations])
    assert response.status_code == 201
    assert response.json() == {"message": "Relations created successfully"}

def test_add_observations():
    # エンティティを作成
    entity = Entity(name="John", entity_type="person", observations=["Speaks English"])
    client.post("/entities", json=[entity.model_dump()])

    # 観察を追加
    observations = [
        Observation(entity_name="John", contents=["Lives in Tokyo"])
    ]
    response = client.post("/observations", json=[obs.model_dump() for obs in observations])
    assert response.status_code == 200
    assert "John" in response.json()["added_observations"]

def test_read_graph():
    # グラフを初期化
    entities = [
        Entity(name="John", entity_type="person", observations=["Speaks English"]),
        Entity(name="Acme", entity_type="organization")
    ]
    client.post("/entities", json=[entity.model_dump() for entity in entities])

    response = client.get("/graph")
    assert response.status_code == 200
    assert "entities" in response.json()
    assert "relations" in response.json()
    assert len(response.json()["entities"]) == 2

def test_search_nodes():
    # テストデータを作成
    entities = [
        Entity(name="John Smith", entity_type="person", observations=["Speaks English"]),
        Entity(name="Jane Doe", entity_type="person", observations=["Speaks Japanese"])
    ]
    client.post("/entities", json=[entity.model_dump() for entity in entities])

    response = client.get("/search", params={"query": "Japanese"})
    assert response.status_code == 200
    assert "Jane Doe" in response.json()["entities"]

def test_open_nodes():
    # テストデータを作成
    entities = [
        Entity(name="John", entity_type="person"),
        Entity(name="Jane", entity_type="person")
    ]
    client.post("/entities", json=[entity.model_dump() for entity in entities])

    response = client.get("/nodes", params={"names": ["John", "Jane"]})
    assert response.status_code == 200
    assert len(response.json()["entities"]) == 2
