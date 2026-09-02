import pytest

from portability.portability import (
    compare_outputs,
    create_model,
)


def test_success_compare_outputs():
    result = compare_outputs(
        "RAG retrieves external knowledge.",
        "RAG retrieves relevant external knowledge.",
    )

    assert result["exact_match"] is False
    assert 0 <= result["similarity"] <= 1
    assert result["length_difference"] >= 0


def test_failure_unsupported_provider():
    with pytest.raises(ValueError):
        create_model("unsupported_provider")