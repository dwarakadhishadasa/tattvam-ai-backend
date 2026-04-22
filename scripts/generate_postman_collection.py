#!/usr/bin/env python3
"""Generate a Postman collection from the FastAPI OpenAPI schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app

OUTPUT_PATH = PROJECT_ROOT / "postman" / "notebooklm-rest-api.postman_collection.json"
COLLECTION_SCHEMA_URL = "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
VARIABLE_MAP = {
    "notebook_id": "notebookId",
    "source_id": "sourceId",
    "task_id": "taskId",
    "artifact_id": "artifactId",
}


def to_variable_name(name: str) -> str:
    if name in VARIABLE_MAP:
        return VARIABLE_MAP[name]

    parts = name.split("_")
    if not parts:
        return name
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def schema_components(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("components", {}).get("schemas", {})


def resolve_schema(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema
    ref_name = schema["$ref"].split("/")[-1]
    return schema_components(spec)[ref_name]


def merge_all_of(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for part in schema.get("allOf", []):
        resolved = normalize_schema(spec, part)
        merged["properties"].update(resolved.get("properties", {}))
        merged["required"].extend(resolved.get("required", []))
    merged["required"] = sorted(set(merged["required"]))
    return merged


def normalize_schema(spec: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    schema = resolve_schema(spec, schema)

    if "allOf" in schema:
        return merge_all_of(spec, schema)

    for key in ("anyOf", "oneOf"):
        if key in schema:
            for option in schema[key]:
                normalized = normalize_schema(spec, option)
                if normalized.get("type") != "null":
                    return normalized
            return normalize_schema(spec, schema[key][0])

    return schema


def example_from_schema(spec: dict[str, Any], schema: dict[str, Any]) -> Any:
    schema = normalize_schema(spec, schema)

    if "default" in schema:
        return schema["default"]
    if "example" in schema:
        return schema["example"]
    if "enum" in schema:
        return schema["enum"][0]

    schema_type = schema.get("type")

    if schema_type == "object" or "properties" in schema or schema.get("additionalProperties"):
        properties = schema.get("properties", {})
        example: dict[str, Any] = {}
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            prop_schema = normalize_schema(spec, prop_schema)
            if prop_name in required or "default" in prop_schema or "example" in prop_schema:
                example[prop_name] = example_from_schema(spec, prop_schema)

        if schema.get("additionalProperties") is True and not example:
            return {}
        return example

    if schema_type == "array":
        item_schema = schema.get("items", {})
        return [example_from_schema(spec, item_schema)]

    if schema_type == "boolean":
        return True
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "string":
        fmt = schema.get("format")
        if fmt == "binary":
            return ""
        if fmt == "uuid":
            return "00000000-0000-0000-0000-000000000000"
        if fmt == "date-time":
            return "2026-01-01T00:00:00Z"
        if fmt == "date":
            return "2026-01-01"
        return ""

    return None


def make_raw_url(path: str) -> str:
    raw = "{{baseUrl}}" + path
    for source, target in VARIABLE_MAP.items():
        raw = raw.replace("{" + source + "}", "{{" + target + "}}")
    return raw


def make_path_segments(path: str) -> list[str]:
    segments: list[str] = []
    for segment in path.lstrip("/").split("/"):
        if segment.startswith("{") and segment.endswith("}"):
            key = segment[1:-1]
            segments.append(":" + to_variable_name(key))
        else:
            segments.append(segment)
    return segments


def build_query_params(spec: dict[str, Any], parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query: list[dict[str, Any]] = []
    for parameter in parameters:
        if parameter.get("in") != "query":
            continue

        schema = normalize_schema(spec, parameter.get("schema", {}))
        value = example_from_schema(spec, schema)
        query.append(
            {
                "key": parameter["name"],
                "value": "" if value is None else str(value),
                "disabled": not parameter.get("required", False),
            }
        )
    return query


def build_headers(content_type: str | None) -> list[dict[str, str]]:
    headers = [{"key": "X-API-Key", "value": "{{apiKey}}", "type": "text"}]
    if content_type == "application/json":
        headers.append({"key": "Content-Type", "value": "application/json", "type": "text"})
    return headers


def build_body(spec: dict[str, Any], operation: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    request_body = operation.get("requestBody")
    if not request_body:
        return None, None

    content = request_body.get("content", {})
    if "application/json" in content:
        schema = content["application/json"].get("schema", {})
        example = example_from_schema(spec, schema)
        return {"mode": "raw", "raw": json.dumps(example, indent=2), "options": {"raw": {"language": "json"}}}, "application/json"

    if "multipart/form-data" in content:
        schema = content["multipart/form-data"].get("schema", {})
        resolved = normalize_schema(spec, schema)
        formdata: list[dict[str, str]] = []
        required = set(resolved.get("required", []))
        for name, prop_schema in resolved.get("properties", {}).items():
            prop_schema = normalize_schema(spec, prop_schema)
            field_type = "file" if prop_schema.get("format") == "binary" else "text"
            value = "" if field_type == "file" else str(example_from_schema(spec, prop_schema) or "")
            entry: dict[str, str | bool] = {"key": name, "type": field_type, "disabled": name not in required}
            if field_type == "text":
                entry["value"] = value
            elif value:
                entry["src"] = value
            formdata.append(entry)  # type: ignore[arg-type]
        return {"mode": "formdata", "formdata": formdata}, None

    return None, None


def folder_name_for_path(path: str) -> str:
    if path == "/health":
        return "Health"
    if "/sources" in path:
        return "Sources"
    if "/chat/" in path:
        return "Chat"
    if "/artifacts" in path:
        return "Artifacts"
    return "Notebooks"


def build_request_item(spec: dict[str, Any], path: str, method: str, operation: dict[str, Any]) -> dict[str, Any]:
    parameters = operation.get("parameters", [])
    body, content_type = build_body(spec, operation)

    item: dict[str, Any] = {
        "name": operation.get("summary") or f"{method.upper()} {path}",
        "request": {
            "method": method.upper(),
            "header": build_headers(content_type),
            "url": {
                "raw": make_raw_url(path),
                "host": ["{{baseUrl}}"],
                "path": make_path_segments(path),
                "query": build_query_params(spec, parameters),
            },
            "description": operation.get("description") or operation.get("summary", ""),
        },
        "response": [],
    }

    if body is not None:
        item["request"]["body"] = body

    return item


def build_collection(spec: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path, methods in spec.get("paths", {}).items():
        folder = folder_name_for_path(path)
        grouped.setdefault(folder, [])
        for method, operation in methods.items():
            grouped[folder].append(build_request_item(spec, path, method, operation))

    folders = [{"name": name, "item": items} for name, items in grouped.items()]

    return {
        "info": {
            "name": spec.get("info", {}).get("title", "API Collection"),
            "_postman_id": "b92d1f0a-b90d-4453-bd6a-0f218739e1fd",
            "description": "Generated from FastAPI OpenAPI schema in app.py",
            "schema": COLLECTION_SCHEMA_URL,
        },
        "item": folders,
        "variable": [
            {"key": "baseUrl", "value": "http://localhost:8000"},
            {"key": "apiKey", "value": ""},
            {"key": "notebookId", "value": "replace-me"},
            {"key": "sourceId", "value": "replace-me"},
            {"key": "taskId", "value": "replace-me"},
            {"key": "artifactId", "value": "replace-me"},
        ],
    }


def main() -> None:
    spec = app.openapi()
    collection = build_collection(spec)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(collection, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
