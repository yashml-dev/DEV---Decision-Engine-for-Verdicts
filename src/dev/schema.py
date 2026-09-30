"""Loads schema.yaml into typed question objects."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

VALID_TYPES = {"choice", "score", "yes_no"}


@dataclass
class Question:
    id: str
    type: str
    options: list = field(default_factory=list)

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(f"Question '{self.id}': type must be one of {VALID_TYPES}")
        if self.type == "choice" and len(self.options) < 2:
            raise ValueError(f"Question '{self.id}': choice needs at least 2 options")


def load_schema(path="configs/schema.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    questions = [Question(**q) for q in raw["questions"]]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in schema")
    return {q.id: q for q in questions}
