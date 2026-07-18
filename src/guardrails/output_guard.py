from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from guardrails.input_guard import GuardResult
from rag.llm_provider import get_generator_llm

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

class ToxicityGuard:
    """Uses a local toxic-bert model to detect toxic generated outputs."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        from transformers import pipeline
        # Cache the pipeline so it's loaded only once
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1 # CPU by default, fine for local inference
        )

    def evaluate(self, answer: str) -> GuardResult:
        try:
            result = self.classifier(answer[:512]) # Truncate to BERT max length
            score = result[0]['score']
            label = result[0]['label']

            is_toxic = (label == 'toxic' and score > self.threshold)

            if is_toxic:
                logger.warning("ToxicityGuard triggered! Score: %.2f", score)
                return GuardResult(passed=False, reason=f"Output flagged as toxic (score: {score:.2f})")

            return GuardResult(passed=True, modified_text=answer)
        except Exception as e:
            logger.error("ToxicityGuard failed to evaluate: %s", e)
            return GuardResult(passed=False, reason=f"Toxicity evaluation failed: {e}")

class TopicGuard:
    """Uses an LLM judge to ensure the answer stays on-topic."""

    def __init__(self, allowed_topics: list[str]):
        self.allowed_topics = allowed_topics

    def evaluate(self, answer: str, question: str) -> GuardResult:
        if not self.allowed_topics:
            return GuardResult(passed=True, modified_text=answer)

        llm = get_generator_llm(temperature=0.0)
        topics_str = ", ".join(self.allowed_topics)

        system_prompt = (
            "You are a strict output guardrail. Your job is to analyze the provided answer "
            f"and determine if it strays from the allowed topics: [{topics_str}].\n"
            "If the answer discusses anything outside these topics, it is OFF-TOPIC.\n"
            "Respond ONLY with valid JSON in this format: {\"is_allowed\": true|false, \"reason\": \"string\"}"
        )

        user_prompt = f"Question: {question}\n\nAnswer: {answer}"

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            cleaned = response.content.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned)
            is_allowed = data.get("is_allowed", True)
            reason = data.get("reason", "No reason provided")

            if not is_allowed:
                logger.warning("TopicGuard triggered: %s", reason)
                return GuardResult(passed=False, reason=f"Off-topic content detected: {reason}")

            return GuardResult(passed=True, modified_text=answer)
        except Exception as e:
            logger.error("TopicGuard failed to evaluate: %s", e)
            return GuardResult(passed=False, reason=f"Topic evaluation failed: {e}")

