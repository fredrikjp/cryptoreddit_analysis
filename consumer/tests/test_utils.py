import pytest
import numpy as np
from ..utils import assert_response_format

def test_valid_response_no_constraints():
    response = '[ [1, 2], [3.5, 4] ]'
    # Should not raise error
    assert_response_format(response)

def test_invalid_json():
    response = '{bad json]'
    with pytest.raises(ValueError, match="Response is not valid JSON format."):
        assert_response_format(response)

def test_response_not_list():
    response = '{"a": 1}'
    with pytest.raises(AssertionError, match="Response should be a list."):
        assert_response_format(response)

def test_row_not_list():
    response = '[1, 2, 3]'
    with pytest.raises(AssertionError, match="Row 0 should be a list."):
        assert_response_format(response)

def test_element_not_number():
    response = '[["a", 2], [3, 4]]'
    with pytest.raises(AssertionError, match="Element 0 in row 0 should be a number."):
        assert_response_format(response)

def test_value_out_of_bounds():
    response = '[[5, 10], [15, 20]]'
    constraints = np.array([[0, 10], [0, 15]])
    with pytest.raises(AssertionError, match="Element 0 in row 1 should be between 0 and 10."):
        assert_response_format(response, value_constraints=constraints)

def test_valid_response_with_constraints():
    response = '[[5, 10], [10, 15]]'
    constraints = np.array([[0, 10], [0, 15]])
    # Should not raise error
    assert_response_format(response, value_constraints=constraints)
