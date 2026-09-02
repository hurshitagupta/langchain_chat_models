import pytest

from langchain_core.runnables import RunnableLambda


def test_success_fallback():
    def failing_primary(value):
        raise ConnectionError("Primary failed")

    def working_fallback(value):
        return "Fallback response"

    primary = RunnableLambda(failing_primary)
    fallback = RunnableLambda(working_fallback)

    resilient = primary.with_fallbacks([fallback])

    result = resilient.invoke("test")

    assert result == "Fallback response"


def test_failure_when_primary_and_fallback_fail():
    def failing_primary(value):
        raise ConnectionError("Primary failed")

    def failing_fallback(value):
        raise RuntimeError("Fallback failed")

    primary = RunnableLambda(failing_primary)
    fallback = RunnableLambda(failing_fallback)

    resilient = primary.with_fallbacks([fallback])

    with pytest.raises(Exception):
        resilient.invoke("test")