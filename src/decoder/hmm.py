"""Helpers for validating and preparing probabilities for HMM decoding."""

import math


def _validate_probability(value, name):
    """Check that a probability is a finite number between zero and one."""

    value = float(value)

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be a finite probability between 0 and 1"
        )

    return value


def _normalize_row(values, name, allow_all_zero=False):
    """Normalize probability scores in a row so they add up to one."""

    checked = [
        _validate_probability(value, name)
        for value in values
    ]

    total = sum(checked)

    if total == 0:
        if allow_all_zero:
            return checked

        raise ValueError(f"{name} cannot contain only zero values")

    return [value / total for value in checked]


def _to_log_probabilities(values):
    """Convert probabilities to log values, keeping zero as negative infinity."""

    return [
        math.log(value) if value > 0 else float("-inf")
        for value in values
    ]


def prepare_hmm_inputs(
    emission_probabilities,
    transition_probabilities,
):
    """
    Check the emission and transition probabilities before HMM decoding.

    emission_probabilities:
        A list of candidate emission scores for each observation.

    transition_probabilities:
        A transition matrix for each pair of consecutive observations.
        Rows correspond to the previous observation's candidates, while
        columns correspond to the current observation's candidates.

    Returns:
        Log-transformed emission and transition probabilities.
    """

    if not emission_probabilities:
        raise ValueError("At least one observation is required")

    emissions = []

    # Validate and normalize the candidate scores for each observation.
    for index, row in enumerate(emission_probabilities):
        if not row:
            raise ValueError(
                f"Observation {index} has no candidate emissions"
            )

        normalized = _normalize_row(
            row, f"emission row {index}"
        )
        emissions.append(_to_log_probabilities(normalized))

    # There should be one transition matrix between each pair of observations.
    if len(transition_probabilities) != len(emissions) - 1:
        raise ValueError(
            "There must be one transition matrix between "
            "each pair of consecutive observations"
        )

    transitions = []

    for index, matrix in enumerate(transition_probabilities):
        previous_count = len(emissions[index])
        current_count = len(emissions[index + 1])

        # Check that the matrix dimensions match the candidate counts.
        if len(matrix) != previous_count:
            raise ValueError(
                f"Transition matrix {index} has an incorrect "
                "number of rows"
            )

        normalized_matrix = []

        for row in matrix:
            if len(row) != current_count:
                raise ValueError(
                    f"Transition matrix {index} has an incorrect "
                    "number of columns"
                )

            normalized_matrix.append(
                _normalize_row(
                    row,
                    f"transition matrix {index}",
                    allow_all_zero=True,
                )
            )

        transitions.append([
            _to_log_probabilities(row)
            for row in normalized_matrix
        ])

    return emissions, transitions
