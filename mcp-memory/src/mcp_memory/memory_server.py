from typing import List, Dict, Optional
from .models import Entity, Relation, Observation
import json
import os

class MemoryServer:
    def __init__(self, file_path: str = "memory.json"):
        self.file_path = file_path
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self._load_data()

    def _load_data(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        self.entities = {name: Entity(**entity) for name, entity in data.get("entities", {}).items()}
                        self.relations = [Relation(**relation) for relation in data.get("relations", [])]
            except json.JSONDecodeError:
                # ファイルが空または不正な形式の場合は初期状態とする
                self._save_data()

    def _save_data(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump({
                "entities": {name: entity.model_dump() for name, entity in self.entities.items()},
                "relations": [relation.model_dump() for relation in self.relations]
            }, f, indent=2)

    def create_entities(self, entities: List[Entity]) -> None:
        for entity in entities:
            if entity.name not in self.entities:
                self.entities[entity.name] = entity
        self._save_data()

    def create_relations(self, relations: List[Relation]) -> None:
        for relation in relations:
            if relation not in self.relations:
                self.relations.append(relation)
        self._save_data()

    def add_observations(self, observations: List[Observation]) -> Dict[str, List[str]]:
        added_observations = {}
        for observation in observations:
            if observation.entity_name in self.entities:
                entity = self.entities[observation.entity_name]
                new_observations = [obs for obs in observation.contents if obs not in entity.observations]
                entity.observations.extend(new_observations)
                added_observations[observation.entity_name] = new_observations
        self._save_data()
        return added_observations

    def delete_entities(self, entity_names: List[str]) -> None:
        for name in entity_names:
            self.entities.pop(name, None)
        self.relations = [r for r in self.relations if r.from_entity not in entity_names and r.to_entity not in entity_names]
        self._save_data()

    def delete_observations(self, deletions: List[Observation]) -> None:
        for deletion in deletions:
            if deletion.entity_name in self.entities:
                entity = self.entities[deletion.entity_name]
                entity.observations = [obs for obs in entity.observations if obs not in deletion.contents]
        self._save_data()

    def delete_relations(self, relations: List[Relation]) -> None:
        self.relations = [r for r in self.relations if r not in relations]
        self._save_data()

    def read_graph(self) -> Dict:
        return {
            "entities": self.entities,
            "relations": self.relations
        }

    def search_nodes(self, query: str) -> Dict:
        matching_entities = {}
        matching_relations = []
        
        for name, entity in self.entities.items():
            if (query.lower() in name.lower() or
                query.lower() in entity.entity_type.lower() or
                any(query.lower() in obs.lower() for obs in entity.observations)):
                matching_entities[name] = entity
                
        for relation in self.relations:
            if (relation.from_entity in matching_entities or
                relation.to_entity in matching_entities):
                matching_relations.append(relation)
                
        return {
            "entities": matching_entities,
            "relations": matching_relations
        }

    def open_nodes(self, names: List[str]) -> Dict:
        entities = {name: self.entities[name] for name in names if name in self.entities}
        relations = [r for r in self.relations if r.from_entity in names and r.to_entity in names]
        return {
            "entities": entities,
            "relations": relations
        }
