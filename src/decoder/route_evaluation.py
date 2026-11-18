
from difflib import SequenceMatcher


def evaluate_route(predicted_route, reference_route):
    """
    Evaluate a predicted road sequence against
    the reference road sequence.
    """

    if predicted_route is None or reference_route is None:
        raise ValueError("Routes cannot be None")

    predicted = list(predicted_route)
    reference = list(reference_route)

    if not predicted or not reference:
        raise ValueError("Routes cannot be empty")

    predicted_set = set(predicted)
    reference_set = set(reference)

    correct = len(predicted_set & reference_set)

    precision = (
        correct / len(predicted_set)
        if predicted_set else 0.0
    )

    recall = (
        correct / len(reference_set)
        if reference_set else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    sequence_similarity = SequenceMatcher(
        None, reference, predicted
    ).ratio()

    return {
        "predicted_length": len(predicted),
        "reference_length": len(reference),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "sequence_similarity": sequence_similarity,
    }
