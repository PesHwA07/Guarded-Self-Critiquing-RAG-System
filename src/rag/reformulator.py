"""Query reformulation module.

Takes the original question and the critic's reasoning for why the previous answer
was insufficient, and produces an improved search query to retrieve better context.
"""

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.llm_provider import get_generator_llm

logger = logging.getLogger(__name__)

REFORMULATOR_PROMPT = """You are an expert at information retrieval and query reformulation.
A previous search for the user's question did not find sufficient information to fully answer it.

Original Question: {question}

The critic provided the following reasoning for why the previous answer was insufficient or lacked context:
{reasoning}

Your task is to write a single, improved search query that is more likely to retrieve the missing information.
- Make it specific and keyword-rich.
- Do NOT include any conversational text, explanations, or quotes.
- Output ONLY the new search query.
"""

def reformulate_query(question: str, critic_reasoning: str) -> str:
    """Reformulate the search query based on critic feedback.

    Args:
        question: The user's original question.
        critic_reasoning: The critic's explanation of what was missing or ungrounded.

    Returns:
        A new, refined search query string.
    """
    llm = get_generator_llm(temperature=0.3)  # Slight temperature for creativity
    prompt = PromptTemplate.from_template(REFORMULATOR_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        new_query = chain.invoke({
            "question": question,
            "reasoning": critic_reasoning,
        })
        new_query = new_query.strip(" '\"\n")
        
        # Fallback if the LLM returned nothing
        if not new_query:
            logger.warning("Reformulator returned empty query. Falling back to original.")
            return question
            
        return new_query
    except Exception as e:
        logger.error("LLM error during query reformulation: %s", e)
        # Fail open: just return the original question so the loop can continue or fallback
        return question
