import random
import torch
import numpy as np


def inspect_word(word, df, text_col, label_col):
    filtered_samples = df[df[text_col].str.contains(word)]
    print(filtered_samples[label_col].value_counts())
    return filtered_samples


def initialize_seeds(seed=42):
    # python RNG
    random.seed(seed)

    # pytorch RNGs
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    np.random.seed(seed)