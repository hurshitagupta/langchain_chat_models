import pytest
from message_construction.message_construction import build_messages, validate_messages
from langchain_core.messages import HumanMessage, AIMessage

def test_success_message_order():
    sequence_a, sequence_b = build_messages()

    assert isinstance(sequence_a[1], HumanMessage)
    assert sequence_a[1].content == "ORANGE"

    assert isinstance(sequence_a[2], AIMessage)
    assert sequence_a[2].content == "PURPLE"

    assert isinstance(sequence_b[1], AIMessage)
    assert sequence_b[1].content == "PURPLE"

    assert isinstance(sequence_b[2], HumanMessage)
    assert sequence_b[2].content == "ORANGE"


def test_failure_message_order():
    with pytest.raises(ValueError):
        validate_messages([])

