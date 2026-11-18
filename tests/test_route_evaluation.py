
import pytest

from src.decoder.route_evaluation import evaluate_route


def test_identical_routes():
    route = [10, 20, 30, 40]

    result = evaluate_route(route, route)

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["sequence_similarity"] == 1.0


def test_partially_matching_routes():
    predicted = [10, 20, 30]
    reference = [10, 20, 40]

    result = evaluate_route(predicted, reference)

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1_score"] == pytest.approx(2 / 3)


def test_completely_different_routes():
    predicted = [10, 20]
    reference = [30, 40]

    result = evaluate_route(predicted, reference)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0


def test_empty_route_rejected():
    with pytest.raises(ValueError):
        evaluate_route([], [10, 20])
