import os
import sys
sys.path.insert(0, os.getcwd())

from dlengine.data_sampler import TrainingSampler

if __name__ == "__main__":
    training_sampler = TrainingSampler(100, shuffle=True, seed=100, infinite=False)
    for _ in range(2):
        for i in training_sampler:
            print(i)
        print("="*20)

