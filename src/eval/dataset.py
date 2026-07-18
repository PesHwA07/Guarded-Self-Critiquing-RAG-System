import json
import random
from typing import List


def load_dataset(path: str, mode: str = "full", smoke_size: int = 10) -> List[dict]:
    """Load the golden dataset from a JSON file.

    Args:
        path: Path to the dataset JSON file.
        mode: 'full' for the entire dataset, 'smoke' for a quick subset.
        smoke_size: Number of items to return in smoke mode.

    Returns:
        A list of dictionary items representing the queries.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if mode == "smoke":
        # Use a fixed seed for deterministic smoke tests
        random.seed(42)
        return random.sample(data, min(smoke_size, len(data)))

    return data
