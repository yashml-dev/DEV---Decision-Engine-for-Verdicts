import pytest
from dev.schema import load_schema, Question
from dev.data import make_synthetic


def test_schema_loads():
    s = load_schema("configs/schema.yaml")
    assert set(s) == {"department", "urgency", "escalate"}
    assert s["department"].type == "choice" and len(s["department"].options) == 4


def test_bad_question_type():
    with pytest.raises(ValueError):
        Question(id="x", type="banana")


def test_choice_needs_options():
    with pytest.raises(ValueError):
        Question(id="x", type="choice", options=["only_one"])


def test_synthetic_data_valid():
    s = load_schema("configs/schema.yaml")
    rows = make_synthetic(50)
    assert len(rows) == 50
    for r in rows:
        assert r["department"] in s["department"].options
        assert 0 <= r["urgency"] <= 1 and r["escalate"] in (0, 1)
