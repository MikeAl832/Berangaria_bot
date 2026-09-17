import pytest

from berangaria.config import _required_price, _required_str


def test_required_str_accepts_nonempty_and_strips():
    assert _required_str({"model": "  foo/bar  "}, "model") == "foo/bar"


@pytest.mark.parametrize("value", [None, "", "   ", 1, True])
def test_required_str_rejects_missing_or_blank(value):
    data = {} if value is None else {"model": value}
    with pytest.raises(ValueError, match="model"):
        _required_str(data, "model")


def test_required_price_accepts_zero_and_int():
    assert _required_price({"price_completion": 0}, "price_completion") == 0.0
    assert _required_price({"price_completion": 1}, "price_completion") == 1.0


@pytest.mark.parametrize("value", [None, -0.01, True, False, "0.10", []])
def test_required_price_rejects_invalid(value):
    data = {} if value is None else {"price_completion": value}
    with pytest.raises(ValueError, match="price_completion"):
        _required_price(data, "price_completion")
