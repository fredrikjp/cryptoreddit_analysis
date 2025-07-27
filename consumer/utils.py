import numpy as np
import json
import typing
from datetime import datetime


def dump_data(data: typing.Dict[str, typing.List], path: str) -> None:
    """
    Saves score_queues-like data (dict of deque of [timestamp, sentiment_dict])
    to a JSON file with proper formatting.
    """
    # Convert deques to lists for JSON serialization
    serializable_data = {
        key: list(value)  # each value is a deque of [timestamp, sentiment_dict]
        for key, value in data.items()
    }

    with open(path, "w") as f:
        json.dump(serializable_data, f, indent=2)


# Assert right language model reposnse format
def assert_response_format(response:str, value_constraints:typing.Union[typing.List, None] = None) -> None:
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON format.")

    assert isinstance(data, list), "Response should be a list."

    for i, row in enumerate(data):
        assert isinstance(row, list), f"Row {i} should be a list."
        for j, value in enumerate(row):
            assert isinstance(value, (int, float)), f"Element {j} in row {i} should be a number."

            if value_constraints is not None:
                assert value_constraints[j,0] <= value <= value_constraints[j,1], f"Element {j} in row {i} should be between {value_constraints[j,0]} and {value_constraints[j,1]}."


# Write Json data to file
def dump_data_old(data: typing.Dict[str, typing.List], file_path: str) -> None:
    json_ready = {}

    for key, value in data.items():
        json_ready[key] =[
            [datetime.now().isoformat(), score] for _, score in value
        ]

    with open(file_path, 'w') as f:
        json.dump(json_ready, f)
