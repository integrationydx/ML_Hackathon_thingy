
import math


def _validate_probability(value, name):
    """Ensure a probability is finite and lies between 0 and 1."""
    value = float(value)

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be a finite probability between 0 and 1"
        )

    return value


def _normalize_row(values, name, allow_all_zero=False):
    """Normalize a list of non-negative probability scores."""
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
    """Convert probabilities to log probabilities safely."""
    return [
        math.log(value) if value > 0 else float("-inf")
        for value in values
    ]


def prepare_hmm_inputs(
    emission_probabilities,
    transition_probabilities,
):
    """
    Validate and prepare probability inputs for HMM decoding.

    emission_probabilities:
        One list of candidate emission scores per observation.

    transition_probabilities:
        One matrix between each consecutive pair of observations.
        Matrix rows represent previous candidates.
        Matrix columns represent current candidates.

    Returns:
        log_emissions, log_transitions
    """

    if not emission_probabilities:
        raise ValueError("At least one observation is required")

    emissions = []

    for index, row in enumerate(emission_probabilities):
        if not row:
            raise ValueError(
                f"Observation {index} has no candidate emissions"
            )

        normalized = _normalize_row(
            row, f"emission row {index}"
        )
        emissions.append(_to_log_probabilities(normalized))

    if len(transition_probabilities) != len(emissions) - 1:
        raise ValueError(
            "There must be one transition matrix between "
            "each pair of consecutive observations"
        )

    transitions = []

    for index, matrix in enumerate(transition_probabilities):
        previous_count = len(emissions[index])
        current_count = len(emissions[index + 1])

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
