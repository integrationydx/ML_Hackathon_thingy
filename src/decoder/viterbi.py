            raise ValueError("Emission values cannot be NaN")

    if len(log_transitions) != num_observations - 1:
        raise ValueError(
            "Incorrect number of transition matrices"
        )

    # Store the best scores found so far and the previous candidate
    # that led to each score.
    scores = [list(log_emissions[0])]
    backpointers = []

    for t in range(1, num_observations):
        previous_scores = scores[-1]
        transition_matrix = log_transitions[t - 1]

        # The transition matrix must match the candidate counts
        # in the previous and current observations.
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

            # Check every possible previous candidate and keep the
            # one that gives the highest cumulative log probability.
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

    # Start backtracking from the candidate with the best final score.
    final_index = max(
        range(len(scores[-1])),
        key=lambda index: scores[-1][index]
    )

    if scores[-1][final_index] == float("-inf"):
        raise ValueError(
            "No valid road sequence exists"
        )

    best_path = [final_index]

    # Follow the stored pointers to reconstruct the full candidate path.
    for pointers in reversed(backpointers):
        previous_index = pointers[best_path[-1]]

        if previous_index is None:
            raise ValueError("No valid path could be reconstructed")

        best_path.append(previous_index)

    best_path.reverse()

    return best_path
