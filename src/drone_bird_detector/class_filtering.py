import torch

def load_included_classes(path: str | None):
    """
    Load included class names from a text file (one per line).
    """
    if path is None:
        return None

    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def build_class_mask(labels, included_classes):
    """
    labels: full labels list INCLUDING background at index 0
    included_classes: set[str]

    Returns BoolTensor aligned with all_scores columns (labels[1:])
    """
    return torch.tensor(
        [cls in included_classes for cls in labels[1:]],
        dtype=torch.bool,
    )

def filter_class_scores(
    fg_scores,
    bg_scores,
    class_mask,
    renormalise=False,
):
    """
    fg_scores: Tensor [N, C]  (foreground only)
    bg_scores: Tensor [N]     (background probability)
    class_mask: BoolTensor [C]
    """

    filtered = fg_scores.clone()
    filtered[:, ~class_mask] = 0.0

    if renormalise:
        # Include background in normalisation
        fg_sum = filtered.sum(dim=1)
        denom = fg_sum + bg_scores

        nonzero = denom > 0
        filtered[nonzero] = filtered[nonzero] / denom[nonzero].unsqueeze(1)
        bg_scores = bg_scores.clone()
        bg_scores[nonzero] = bg_scores[nonzero] / denom[nonzero]

    return filtered, bg_scores