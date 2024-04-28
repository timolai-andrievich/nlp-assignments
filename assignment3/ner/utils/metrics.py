from .types import Entity

def char_wise_f1_score_macro(
    inputs: list[Entity],
    target: list[Entity],
    n_labels: int,
) -> float:
    """Calculates macro F1 score character-wise.
    
    Args:
        inputs (list[Entity]): Predicted values.
        target (list[Entity]): True values.
        n_labels (int): Number of labels in the dataset.
    
    Returns:
        float: F1 score.
    """
    if not inputs or not target:
        return 0.0
    # Find the total length of the text
    text_len = max(end for _, end, _ in inputs + target) + 1
    labels = sorted({label for _, _, label in inputs + target})
    label2id = {label: idx for idx, label in enumerate(labels)}
    label2id['O'] = -1
    true_positives = {label: 0 for label in label2id.values()}
    false_positives = {label: 0 for label in label2id.values()}
    false_negatives = {label: 0 for label in label2id.values()}
    # Construct character-wise predictions
    char_labels_inputs = [-1 for _ in range(text_len)]
    for start, end, label in inputs:
        label = label2id[label]
        for i in range(start, end + 1):
            char_labels_inputs[i] = label
    # Construct character-wise target
    char_labels_target = [-1 for _ in range(text_len)]
    for start, end, label in target:
        label = label2id[label]
        for i in range(start, end + 1):
            char_labels_target[i] = label
    # Count errors and true predictions
    for true_label, pred_label in zip(char_labels_target, char_labels_inputs):
        if true_label == pred_label:
            true_positives[true_label] += 1
        else:
            false_positives[pred_label] += 1
            false_negatives[true_label] += 1
    # Calculate the f1 score for each label
    f1_score_per_label = {}
    for label in label2id.values():
        numerator = 2 * true_positives[label]
        denominator = 2 * true_positives[label] + false_negatives[label] + false_positives[label]
        if denominator == 0:
            res = 1
        else:
            res = numerator / denominator
        f1_score_per_label[label] = res
    avg_score = (sum(f1_score_per_label.values()) + n_labels -
                 len(f1_score_per_label)) / n_labels
    return avg_score