import numpy as np
from typing import List, Dict, Callable


def show_token_distribution(
        data: List[float], bin_spacing: int) -> None:
    """Show the probability distribution of all token"""
    min_val = min(data)
    max_val = max(data)
    bins = np.arange(min_val, max_val, bin_spacing)
    counts, _ = np.histogram(data, bins)
    for b, c in zip(bins, counts):
        print(f"{b}: {c}")


def is_valid_num(val: str) -> bool:
    """Log the top-k token candidates and selected token for debugging."""
    try:
        float(val)
        return True
    except ValueError:
        return False


def char_freq(data: Dict) -> Dict[str, int]:
    """Log the top-k token candidates and selected token
    for debugging."""
    char_frequency: Dict[str, int] = {}
    for key, _ in data.items():
        for c in key:
            if c in char_frequency:
                char_frequency[c] += 1
            else:
                char_frequency[c] = 1
    return char_frequency


def show_top_logits(decode: Callable,
                    logits_np: List[float],
                    max_logits_idx: int,
                    top: int) -> None:
    """Log the top-k token candidates and selected token for debugging."""

    sorted_idx = np.argsort(logits_np)
    tops = sorted_idx[-top:]
    tokens_with_prob = ""
    for token in tops:
        # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
        tokens_with_prob += \
            f"{decode([token])}({round(logits_np[token], 2)}),"
    tokens_with_prob += (
        f"\033[92mSelected token: {decode([max_logits_idx])}\033[0m")
    print(tokens_with_prob)
