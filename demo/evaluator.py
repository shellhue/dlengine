
import torch
import itertools
import numpy as np

from dlengine import comm
from dlengine.evaluator import DatasetEvaluator


class ClassificationEvaluator(DatasetEvaluator):
    def __init__(self, distributed=True, classes=None):
        self._preds = []
        self._gts = []
        self._distributed = distributed
        self._classes = classes
        self._report = ""
        self._report_dict = {}

    def reset(self):
        self._preds = []
        self._gts = []

    def process(self, inputs, outputs: torch.Tensor):
        _, labels = inputs
        _, preds = torch.max(outputs, 1)
        labels = labels.cpu().detach().numpy().tolist()
        preds = preds.cpu().detach().numpy().tolist()
        self._preds.extend(preds)
        self._gts.extend(labels)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            preds = comm.gather(self._preds, dst=0)
            preds = list(itertools.chain(*preds))

            gts = comm.gather(self._gts, dst=0)
            gts = list(itertools.chain(*gts))

            if not comm.is_main_process():
                return {}
        else:
            preds = self._preds
            gts = self._gts

        if len(preds) == 0:
            print("[ClassificationEvaluator] Did not receive valid predictions.")
            return {}
        p = (np.asarray(gts) == np.asarray(preds)).sum() * 1.0 / len(gts)
        return {
            "precision": p
        }

