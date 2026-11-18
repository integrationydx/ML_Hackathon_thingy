
import pytest

from src.decoder.viterbi import viterbi_decode


def test_viterbi_finds_best_path():
    emissions = [
        [-0.1, -2.0],
        [-2.0, -0.1],
        [-0.1, -2.0],
    ]

    transitions = [
        [
            [-2.0, -0.1],
            [-0.1, -2.0],
        ],
        [
            [-2.0, -0.1],
            [-0.1, -2.0],
        ],
    ]

    result = viterbi_decode(emissions, transitions)

    assert result == [0, 1, 0]


def test_viterbi_rejects_empty_input():
    with pytest.raises(ValueError):
        viterbi_decode([], [])


def test_viterbi_rejects_invalid_transition_count():
    emissions = [[-0.1, -0.2], [-0.3, -0.4]]

    with pytest.raises(ValueError):
        viterbi_decode(emissions, [])


def test_viterbi_rejects_impossible_path():
    emissions = [[-0.1], [-0.2]]

    transitions = [
        [[float("-inf")]]
    ]

    with pytest.raises(ValueError):
        viterbi_decode(emissions, transitions)

