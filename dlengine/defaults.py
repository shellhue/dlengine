# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import logging
import os
import torch
from typing import Any
from torch.nn.parallel import DistributedDataParallel

from .utils.checkpoint import Checkpointer
from .utils.collect_env import collect_env_info
from .utils.env import TORCH_VERSION, seed_all_rng
from .utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from .utils.file_io import PathManager
from .utils.log import setup_logger
from .utils.testing import print_csv_format
from .evaluator import inference_on_dataset
from .train_loop import SimpleTrainer, TrainerBase
from . import hooks, comm

__all__ = ["default_setup", "DefaultTrainer"]


def default_setup(output_dir, project_name, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the engine logger
    2. Log basic information about environment, cmdline arguments

    Args:
        output_dir (str): the directory to save output info
        args (argparse.NameSpace): the command line arguments to be logged
    """
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name=project_name)
    setup_logger(output_dir, distributed_rank=rank, name="dlengine")
    logger = logging.getLogger(__name__)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    # if not (hasattr(args, "eval_only") and args.eval_only):
    #     torch.backends.cudnn.benchmark = False


class DefaultTrainer(TrainerBase):
    """
    A trainer with default training logic. It does the following:

    1. Create a :class:`SimpleTrainer` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self,
                 cfg: Any,
                 output_dir: str,
                 project_name: str,
                 num_epoch: int,
                 save_model_every_n_epoch: int = 1,
                 log_every_n_iter: int = 20,
                 test_every_n_iter: int = 1):
        """
        Args:
            cfg (Any): the cfg object each project use to build everything, \
                 default trainer does not use cfg in internal, so each project \
                 can pass everything you like
            output_dir (str): the output directory for saving
            project_name (str): the name of the main project using this training engine
            num_epoch (int): the total training epoch
            save_model_every_n_epoch (int): the model saving frequency.
            log_every_n_iter (int): the loging frequency.
            test_every_n_iter (int): the testing frequency.

        """
        super().__init__()
        logger = logging.getLogger(__name__)
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger(name=project_name)

        self.output_dir = output_dir
        self.project_name = project_name
        self.num_epoch = num_epoch
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.log_every_n_iter = log_every_n_iter
        self.test_every_n_iter = test_every_n_iter
        self.start_epoch = 0
        self.end_epoch = num_epoch
        self.cfg = cfg

        # Assume these objects must be constructed in this order.
        model = self.build_model(self.cfg)
        optimizer = self.build_optimizer(self.cfg, model)
        batches_per_epoch, data_loader = self.build_train_loader(self.cfg)
        self.iters_per_epoch = batches_per_epoch
        self.end_iter = self.end_epoch * self.iters_per_epoch
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(self.cfg, self.end_iter, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            output_dir,
            save_to_disk=comm.is_main_process(),
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, weights, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(weights, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            self.start_epoch = self.start_iter // self.iters_per_epoch
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(self.checkpointer, self.save_model_every_n_epoch * self.iters_per_epoch))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(self.test_every_n_iter * self.iters_per_epoch, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=self.log_every_n_iter))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.end_iter),
            JSONWriter(os.path.join(self.output_dir, "metrics.json")),
            TensorboardXWriter(self.output_dir),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_epoch, self.end_epoch, self.iters_per_epoch)
        # if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
        #     assert hasattr(
        #         self, "_last_eval_results"
        #     ), "No evaluation results obtained during training!"
        #     verify_results(self.cfg, self._last_eval_results)
        #     return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg: Any) -> torch.nn.Module:
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.

        Returns:
            torch.nn.Module: the input of the returned model will be the output of dataloader returned by `build_train_loader` or `build_test_loader`

        Should be overwritten to provide a torch.nn.Module.
        """
        raise NotImplementedError

    @classmethod
    def build_optimizer(cls, cfg: Any, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.
            model (torch.nn.Module): the model returned by `build_model`
        Returns:
            torch.optim.Optimizer:

        Should be overwritten to provide a torch.optim.Optimizer.
        """
        raise NotImplementedError

    @classmethod
    def build_lr_scheduler(cls, cfg: Any, end_iter: int,
                           optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.
            end_iter (int): the end iteration of training.
            optimizer (torch.optim.Optimizer): the optimizer returned by `build_optimizer`

        Should be overwritten to provide a torch.optim.lr_scheduler.
        """
        raise NotImplementedError

    @classmethod
    def build_train_loader(cls, cfg: Any) -> (int, torch.utils.data.DataLoader):
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.

        Returns:
            tuple (int, torch.utils.data.DataLoader): first is batches_per_epoch, second is iterable dataloader. The returned dataloader should be infinit.

        Should be overwritten to provide (batches_per_epoch, torch.utils.data.DataLoader).
        """
        raise NotImplementedError

    @classmethod
    def build_test_loader(cls, cfg: Any) -> torch.utils.data.DataLoader:
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.

        Returns:
            torch.utils.data.DataLoader:

        Should be overwritten to provide torch.utils.data.DataLoader.
        """
        raise NotImplementedError

    @classmethod
    def build_evaluator(cls, cfg: Any):
        """
        Args:
            cfg (Any): the config object passed in when trainer initialized. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.

        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
            If you want DefaultTrainer to automatically run evaluation,
            please implement `build_evaluator()` in subclasses (see train_net.py for example).
            Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
        )

    @classmethod
    def test(cls, cfg: Any, model: torch.nn.Module) -> dict:
        """
        Args:
            cfg (Any): the config object. It can be a CfgNode, a argparse.NameSpace, a dict, or anything else you like.
            model (nn.Module):

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        data_loader = cls.build_test_loader(cfg)
        # When evaluators are passed in as arguments,
        # implicitly assume that evaluators can be created before data_loader.
        try:
            evaluator = cls.build_evaluator(cfg)
        except NotImplementedError:
            logger.warn(
                "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                "or implement its `build_evaluator` method."
            )
        results = inference_on_dataset(model, data_loader, evaluator)
        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            logger.info("Evaluation results in csv format:")
            print_csv_format(results)

        return results


# Access basic attributes from the underlying trainer
for _attr in ["model", "data_loader", "optimizer"]:
    setattr(
        DefaultTrainer,
        _attr,
        property(
            # getter
            lambda self, x=_attr: getattr(self._trainer, x),
            # setter
            lambda self, value, x=_attr: setattr(self._trainer, x, value),
        ),
    )
