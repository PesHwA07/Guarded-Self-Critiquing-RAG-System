from pydantic import BaseModel

from guardrails.output_guard import SchemaGuard, TopicGuard, ToxicityGuard


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

def test_toxicity_guard_clean():
    guard = ToxicityGuard()
    result = guard.evaluate("To make a POST request, use the requests.post() method.")
    assert result.passed

def test_toxicity_guard_toxic():
    guard = ToxicityGuard()
    # A clearly toxic phrase for testing purposes
    result = guard.evaluate("You are a stupid idiot and I hate you.")
    assert not result.passed
    assert "toxic" in result.reason.lower()

def test_topic_guard_on_topic():
    guard = TopicGuard(allowed_topics=["Python programming", "requests library"])
    answer = "To make a POST request in Python, you can use requests.post()."
    result = guard.evaluate(answer, "How do I make a POST request?")
    assert result.passed

def test_topic_guard_off_topic():
    guard = TopicGuard(allowed_topics=["Python programming", "requests library"])
    answer = "The best way to cook a steak is medium rare on a charcoal grill."
    result = guard.evaluate(answer, "How do I cook a steak?")
    assert not result.passed
    assert "off-topic" in result.reason.lower()
