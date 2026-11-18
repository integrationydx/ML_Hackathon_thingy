
import math


def viterbi_decode(log_emissions, log_transitions):
    """
    Find the most likely sequence of road candidates
    using the Viterbi algorithm.

    log_emissions:
        Emission log probabilities for each observation.

    log_transitions:
        Transition log probabilities between observations.

    Returns:
        List of selected candidate indices.
    """

    if not log_emissions:
        raise ValueError("At least one observation is required")

    num_observations = len(log_emissions)

    for index, row in enumerate(log_emissions):
        if not row:
            raise ValueError(
                f"Observation {index} has no candidates"
            )

        if any(math.isnan(value) for value in row):
            raise ValueError("Emission values cannot be NaN")

    if len(log_transitions) != num_observations - 1:
        raise ValueError(
            "Incorrect number of transition matrices"
        )

    scores = [list(log_emissions[0])]
    backpointers = []

    for t in range(1, num_observations):
        previous_scores = scores[-1]
        transition_matrix = log_transitions[t - 1]

        if len(transition_matrix) != len(previous_scores):
            raise ValueError("Incorrect transition matrix rows")

        if any(
            len(row) != len(log_emissions[t])
            for row in transition_matrix
        ):
            raise ValueError("Incorrect transition matrix columns")

        current_scores = []
        current_backpointers = []

        for current_index, emission in enumerate(log_emissions[t]):
            best_score = float("-inf")
            best_previous = None

            for previous_index, previous_score in enumerate(
                previous_scores
            ):
                transition = transition_matrix[previous_index][
                    current_index
                ]

                candidate_score = (
                    previous_score + transition + emission
                )

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = previous_index

            current_scores.append(best_score)
            current_backpointers.append(best_previous)

        scores.append(current_scores)
        backpointers.append(current_backpointers)

    final_index = max(
        range(len(scores[-1])),
        key=lambda index: scores[-1][index]
    )

    if scores[-1][final_index] == float("-inf"):
        raise ValueError(
            "No valid road sequence exists"
        )

    best_path = [final_index]

    for pointers in reversed(backpointers):
        previous_index = pointers[best_path[-1]]

        if previous_index is None:
            raise ValueError("No valid path could be reconstructed")

        best_path.append(previous_index)

    best_path.reverse()

    return best_path
