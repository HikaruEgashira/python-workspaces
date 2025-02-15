from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict
from .models import Entity, Relation, Observation
from .memory_server import MemoryServer

app = FastAPI(title="MCP Memory Server")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.server = MemoryServer("memory.json")
    yield
    # クリーンアップが必要な場合はここに記述

app = FastAPI(title="MCP Memory Server", lifespan=lifespan)

def get_server() -> MemoryServer:
    return app.state.server

@app.post("/entities", status_code=201)
async def create_entities(entities: List[Entity]) -> Dict[str, str]:
    get_server().create_entities(entities)
    return {"message": "Entities created successfully"}

@app.post("/relations", status_code=201)
async def create_relations(relations: List[Relation]) -> Dict[str, str]:
    get_server().create_relations(relations)
    return {"message": "Relations created successfully"}

@app.post("/observations")
async def add_observations(observations: List[Observation]) -> Dict[str, Dict[str, List[str]]]:
    added = get_server().add_observations(observations)
    return {"added_observations": added}

@app.delete("/entities")
async def delete_entities(entity_names: List[str]) -> Dict[str, str]:
    get_server().delete_entities(entity_names)
    return {"message": "Entities deleted successfully"}

@app.delete("/observations")
async def delete_observations(deletions: List[Observation]) -> Dict[str, str]:
    get_server().delete_observations(deletions)
    return {"message": "Observations deleted successfully"}

@app.delete("/relations")
async def delete_relations(relations: List[Relation]) -> Dict[str, str]:
    get_server().delete_relations(relations)
    return {"message": "Relations deleted successfully"}

@app.get("/graph")
async def read_graph() -> Dict:
    return get_server().read_graph()

@app.get("/search")
async def search_nodes(query: str) -> Dict:
    return get_server().search_nodes(query)

from fastapi import Query

@app.get("/nodes")
async def open_nodes(names: List[str] = Query(None)) -> Dict:
    if not names:
        return {"entities": {}, "relations": []}
    return get_server().open_nodes(names)
