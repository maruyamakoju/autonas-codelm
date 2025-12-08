"""
Tests for calculator module
Mix of passing and failing tests for demonstration
"""

import pytest
from calculator import add, subtract, multiply, divide


def test_add():
    """Test addition"""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    """Test subtraction"""
    assert subtract(5, 3) == 2
    assert subtract(0, 0) == 0
    assert subtract(-1, -1) == 0


def test_multiply():
    """Test multiplication"""
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 100) == 0


def test_divide():
    """Test division"""
    assert divide(6, 2) == 3
    assert divide(5, 2) == 2.5
    assert divide(-10, 2) == -5


def test_divide_by_zero():
    """Test division by zero raises error"""
    with pytest.raises(ValueError):
        divide(10, 0)


# Intentionally failing test (will be fixed by Claude)
def test_add_intentional_fail():
    """This test will fail intentionally"""
    # Bug: wrong expected value
    assert add(10, 5) == 20  # Should be 15
