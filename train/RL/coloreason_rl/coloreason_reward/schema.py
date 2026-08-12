import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class SchemaRegistry:
    def __init__(self, schema_dir: str | Path):
        self.schema_dir = Path(schema_dir).resolve()

    @lru_cache(maxsize=None)
    def _validator(self, schema_id: str) -> Draft202012Validator:
        if not schema_id or Path(schema_id).name != schema_id:
            raise ValueError(f"invalid schema_id: {schema_id!r}")
        path = self.schema_dir / f"{schema_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"schema not found: {path}")
        with path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    def validate(self, instance: dict[str, Any], schema_id: str) -> bool:
        return self._validator(schema_id).is_valid(instance)
