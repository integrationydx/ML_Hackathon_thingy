
import math
import pytest

from src.decoder.hmm import prepare_hmm_inputs


def test_prepare_hmm_inputs():
    emissions = [
        [0.8, 0.2],
        [0.3, 0.7],
    ]

    transitions = [
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    ]

    log_emissions, log_transitions = prepare_hmm_inputs(
        emissions, transitions
    )

    assert len(log_emissions) == 2
    assert len(log_transitions) == 1

    assert math.isclose(
        math.exp(log_emissions[0][0]), 0.8
    )

    assert math.isclose(
        math.exp(log_transitions[0][0][0]), 0.9
    )


def test_zero_transition_is_negative_infinity():
    emissions = [[0.8, 0.2], [0.5, 0.5]]

    transitions = [
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ]

    _, log_transitions = prepare_hmm_inputs(
        emissions, transitions
    )

    assert log_transitions[0][0][1] == float("-inf")


def test_rejects_invalid_probability():
    with pytest.raises(ValueError):
        prepare_hmm_inputs(
            [[0.8, 1.2]],
            [],
        )


def test_rejects_wrong_transition_dimensions():
    with pytest.raises(ValueError):
        prepare_hmm_inputs(
            [[0.8, 0.2], [0.5, 0.5]],
            [[[0.9], [0.1]]],
        )


def test_rejects_empty_emission_row():
    with pytest.raises(ValueError):
        prepare_hmm_inputs(
            [[0.8, 0.2], []],
            [[]],
        )
