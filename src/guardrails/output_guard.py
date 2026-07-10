from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from guardrails.input_guard import GuardResult

logger = logging.getLogger(__name__)

class SchemaGuard:
    """Validates that the generator's raw output matches a specified Pydantic schema."""

    def __init__(self, schema_model: type[BaseModel]):
        self.schema_model = schema_model

    def evaluate(self, raw_output: str) -> GuardResult:
        # Try to extract JSON from markdown fences if present
        cleaned = raw_output
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
            # Validate against Pydantic schema
            self.schema_model(**data)
            return GuardResult(passed=True, modified_text=raw_output)
        except json.JSONDecodeError as e:
            logger.warning("SchemaGuard failed: Invalid JSON (%s)", e)
            return GuardResult(
                passed=False,
                reason=f"Output is not valid JSON: {e}"
            )
        except ValidationError as e:
            logger.warning("SchemaGuard failed: Pydantic validation error (%s)", e)
            return GuardResult(
                passed=False,
                reason=f"Output does not match required schema: {e}"
            )
