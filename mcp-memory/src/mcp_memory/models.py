from typing import List
from pydantic import BaseModel

class Entity(BaseModel):
    name: str
    entity_type: str
    observations: List[str] = []

class Relation(BaseModel):
    from_entity: str
    to_entity: str
    relation_type: str

class Observation(BaseModel):
    entity_name: str
    contents: List[str]
