from . import comm
import torch
from torch.utils.data.dataloader import DataLoader
from .data_sampler import TrainingSampler, InferenceSampler
from torch.nn.parallel import DistributedDataParallel
import os
from distutils.dir_util import copy_tree


class DistBridger(object):
    @classmethod
    def to_dist_dataloader(cls, dataloader: DataLoader, is_train=True, infinite=False):
        if comm.get_world_size() == 1:
            return dataloader
        batch_size = dataloader.batch_size
        assert batch_size % comm.get_world_size() == 0, "batch size should be integer multiple of world size"
        batch_size = batch_size // comm.get_world_size()
        num_workers = dataloader.num_workers
        dataset = dataloader.dataset
        collate_fn = dataloader.collate_fn
        if is_train:
            sampler = TrainingSampler(len(dataset), shuffle=True, seed=583, infinite=infinite)
        else:
            sampler = InferenceSampler(len(dataset), split=False)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, True)
        return DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=num_workers,
                          pin_memory=True,
                          collate_fn=collate_fn)

    @classmethod
    def to_dist_model(cls, model):
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        return model

    @classmethod
    def dist_inference(cls, inputs, model):
        return model(inputs)

    @classmethod
    def print_on_main(cls, *args, **kwargs):
        if comm.is_main_process():
            print(*args, **kwargs)

    @classmethod
    def makedirs_on_main(cls, dirs):
        if comm.is_main_process():
            if not os.path.exists(dirs):
                os.makedirs(dirs)
        comm.synchronize()

    @classmethod
    def copy_tree_on_main(cls, src, dst):
        if comm.is_main_process():
            if not os.path.exists(dst):
                os.makedirs(dst)
            copy_tree(src, dst)
        comm.synchronize()

    @classmethod
    def save_state_dict_on_main(cls, state_dict, output):
        if comm.is_main_process():
            torch.save(state_dict, output, _use_new_zipfile_serialization=False)
        comm.synchronize()
