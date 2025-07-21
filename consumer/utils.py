import numpy as np
import json
import typing

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
