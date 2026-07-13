from pathlib import Path
import yaml
from pydantic import BaseModel, Field

class PIIPolicy(BaseModel):
    enabled: bool = True
    action: str = "redact"

class InjectionPolicy(BaseModel):
    enabled: bool = True

class ToxicityPolicy(BaseModel):
    enabled: bool = True
    threshold: float = 0.5

class TopicsPolicy(BaseModel):
    enabled: bool = True
    allowed: list[str] = Field(default_factory=list)

class GuardrailPolicies(BaseModel):
    pii: PIIPolicy = Field(default_factory=PIIPolicy)
    injection: InjectionPolicy = Field(default_factory=InjectionPolicy)
    toxicity: ToxicityPolicy = Field(default_factory=ToxicityPolicy)
    topics: TopicsPolicy = Field(default_factory=TopicsPolicy)

def load_policies(path: str | Path = "data/policies.yaml") -> GuardrailPolicies:
    """Loads guardrail policies from a YAML file."""
    policy_path = Path(path)
    if not policy_path.exists():
        # Fall back to root path if running from deep directory during tests
        # Or just return defaults if truly missing
        root_policy_path = Path(__file__).parent.parent.parent / "data" / "policies.yaml"
        if root_policy_path.exists():
            policy_path = root_policy_path
        else:
            return GuardrailPolicies()
            
    with open(policy_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return GuardrailPolicies(**(data or {}))
