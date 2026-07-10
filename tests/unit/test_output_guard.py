import pytest
from pydantic import BaseModel
from guardrails.output_guard import SchemaGuard

class MockResponseSchema(BaseModel):
    answer: str
    sources_used: list[int]
    confidence: str

def test_schema_guard_valid():
    guard = SchemaGuard(MockResponseSchema)
    raw_output = '{"answer": "Here is the answer.", "sources_used": [1, 2], "confidence": "high"}'
    result = guard.evaluate(raw_output)
    assert result.passed

def test_schema_guard_invalid_json():
    guard = SchemaGuard(MockResponseSchema)
    raw_output = '{"answer": "Here is the answer", missing quotes and stuff}'
    result = guard.evaluate(raw_output)
    assert not result.passed
    assert "not valid JSON" in result.reason

def test_schema_guard_missing_fields():
    guard = SchemaGuard(MockResponseSchema)
    raw_output = '{"answer": "Here is the answer."}'
    result = guard.evaluate(raw_output)
    assert not result.passed
    assert "required schema" in result.reason

def test_schema_guard_markdown_fences():
    guard = SchemaGuard(MockResponseSchema)
    raw_output = '''```json\n{"answer": "Here is the answer.", "sources_used": [1], "confidence": "high"}\n```'''
    result = guard.evaluate(raw_output)
    assert result.passed
