import math
import random
from typing import List
from torch.utils.data import Sampler


class RepeatFactorSampler(Sampler[int]):
    """LVIS-style Repeat Factor Sampler.
    Given per-index repeat factors r_i, yields each index i ~ ceil(r_i) times in expectation.
    """
    def __init__(self, indices: List[int], repeat_factors: List[float], shuffle: bool = True):
        self.indices = list(indices)
        self.r = list(repeat_factors)
        assert len(self.indices) == len(self.r)
        self.shuffle = shuffle

    def __iter__(self):
        out = []
        for idx, ri in zip(self.indices, self.r):
            m = int(math.floor(ri))
            frac = ri - m
            reps = m + (1 if random.random() < frac else 0)
            if reps > 0:
                out.extend([idx] * reps)
        if self.shuffle:
            random.shuffle(out)
        return iter(out)

    def __len__(self):
        # Approximate upper bound; DataLoader uses for prefetching.
        return int(sum(max(1, int(math.floor(ri)) + (1 if (ri - math.floor(ri)) > 0 else 0)) for ri in self.r))

