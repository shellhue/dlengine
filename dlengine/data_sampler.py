# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import itertools
from typing import Optional

import numpy as np
from torch.utils.data import Sampler

from . import comm


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        infinite: bool = False,
        drop_last: bool = True,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
            infinite (bool): whether to generate infinite indices or not.
            drop_last (bool): whether to drop last or not when size is not interger multiple of world size.
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._infinite = infinite
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._epoch = 0
        self._drop_last = drop_last

    def __len__(self):
        assert not self._infinite, "when in infinite mode, __len__ should not be called"
        base = self._size // self._world_size
        if self._size % self._world_size > self._rank and not self._drop_last:
            base += 1

        return base

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._indices(), start, None, self._world_size)

    def _indices(self):
        keep = self._size
        if self._drop_last:
            keep = self._size // self._world_size * self._world_size
        if self._infinite:
            np.random.seed(self._seed)
            while True:
                if self._shuffle:
                    yield from np.random.permutation(self._size)[:keep]
                else:
                    yield from np.arange(self._size)[:keep]
        else:
            np.random.seed(self._seed + self._epoch)
            if self._shuffle:
                yield from np.random.permutation(self._size)[:keep]
            else:
                yield from np.arange(self._size)[:keep]
        self._epoch += 1


class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int, split: bool = False):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            split (bool): the flag indicating whether indices should be splited for different gpu
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        if split:
            shard_size = (self._size - 1) // self._world_size + 1
            begin = shard_size * self._rank
            end = min(shard_size * (self._rank + 1), self._size)
            self._local_indices = range(begin, end)
        else:
            self._local_indices = range(0, self._size)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
