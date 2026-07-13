import pytest
from pathlib import Path
from guardrails.policy_engine import load_policies

def test_load_policies():
    policies = load_policies()
    
    assert policies.pii.enabled is True
    assert policies.pii.action == "redact"
    
    assert policies.injection.enabled is True
    
    assert policies.toxicity.enabled is True
    assert policies.toxicity.threshold == 0.5
    
    assert policies.topics.enabled is True
    assert "Python programming" in policies.topics.allowed
    assert "software engineering" in policies.topics.allowed
