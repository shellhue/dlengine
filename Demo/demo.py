from __future__ import print_function
from __future__ import division
import torch
import os
import argparse
import torch.optim as optim

from dlengine.defaults import DefaultTrainer, default_setup
from dlengine.lr_scheduler import WarmupMultiStepLR
from dlengine.data_sampler import TrainingSampler, InferenceSampler
from dlengine.utils.checkpoint import Checkpointer
from dlengine.launch import launch
from dlengine.bridger import DistBridger

from .model import Classifier
from .evaluator import ClassificationEvaluator


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg, cfg.output_dir, "classifier", cfg.num_epoch, cfg.log_every_n_epoch)

    @classmethod
    def build_model(cls, cfg) -> torch.nn.Module:
        # Detect if we have a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Classifier(model_name=cfg.model_name,
                           num_classes=cfg.num_classes,
                           device=device,
                           feature_extract=cfg.feature_extract,
                           use_pretrained=cfg.use_pretrained)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        params_to_update = model.parameters()
        return optim.SGD(params_to_update, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    @classmethod
    def build_lr_scheduler(cls, cfg, end_iter, optimizer):
        milestones = [int(0.25 * end_iter), int(0.5 * end_iter), int(0.75 * end_iter)]
        return WarmupMultiStepLR(optimizer, milestones)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = build_dataset(cfg.task_type,
                                phase="train",
                                transforms=build_transforms("train", cfg.input_size),
                                root_dir=cfg.data_dir,
                                classes=cfg.classes)
        imgs_per_gpu = cfg.batch_size // cfg.num_gpu
        batches_per_epoch = len(dataset) // cfg.batch_size
        sampler = TrainingSampler(len(dataset), shuffle=True, seed=583, infinite=True)
        batch_sampler = torch.utils.data.BatchSampler(sampler, imgs_per_gpu, True)
        collate_fn = None
        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        return batches_per_epoch, torch.utils.data.DataLoader(dataset,
                                                              batch_sampler=batch_sampler,
                                                              num_workers=16,
                                                              pin_memory=True,
                                                              collate_fn=collate_fn)

    @classmethod
    def build_test_loader(cls, cfg):
        dataset = build_dataset(cfg.task_type,
                                phase=cfg.eval_phase,
                                transforms=build_transforms("val", cfg.input_size),
                                root_dir=cfg.data_dir,
                                classes=cfg.classes)
        imgs_per_gpu = cfg.batch_size // cfg.num_gpu
        sampler = InferenceSampler(len(dataset), split=True)
        batch_sampler = torch.utils.data.BatchSampler(sampler, imgs_per_gpu, False)
        return torch.utils.data.DataLoader(dataset,
                                           batch_sampler=batch_sampler,
                                           num_workers=16,
                                           pin_memory=True,
                                           collate_fn=None)

    @classmethod
    def build_evaluator(cls, cfg):
        classes = loader.load_lines_from_file(cfg.classes)
        return ClassificationEvaluator(classes=classes)


def main(args):
    working_dir = os.path.join("./working_dir", args.exp_name)
    # copy data to local when training on remote server
    if args.polyaxon:
        from polyaxon_client.tracking import get_outputs_path, get_data_paths
        dataroot = os.path.join(get_data_paths()["ceph"], args.data_dir)
        polyaxon_output = get_outputs_path()
        args.data_dir = dataroot
        working_dir = os.path.join(polyaxon_output, "working_dir")
        args.classes = os.path.join(get_data_paths()["ceph"], args.classes)
    DistBridger.print_on_main(args.data_dir)
    DistBridger.print_on_main(args.classes)

    args.output_dir = working_dir

    default_setup(args.output_dir, "classifier", args)

    if args.eval_only:
        model = Trainer.build_model(args)
        Checkpointer(model).load(args.weights)  # load trained model

        res = Trainer.test(args, model)
        return res

    trainer = Trainer(args)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train params.')
    parser.add_argument('--task_type', type=str, default="classification",
                        help='possible value is classification, relationship)')
    parser.add_argument('--data_dir', type=str, help='datasets root directory)')
    parser.add_argument('--classes', type=str, default="", help='class name list')
    parser.add_argument('--weights', type=str, default="", help='trained weights for testing')
    parser.add_argument('--num_classes', type=int, help='class number')
    parser.add_argument('--num_gpu', type=int, default=1, help='number of gpus for trainning')
    parser.add_argument('--input_size', type=int, default=224, help='number of classes')
    parser.add_argument('--exp_name', type=str, default="", help='the name used to creat working directory')
    parser.add_argument('--lr', type=float, default=0.001, help='datasets root directory)')
    parser.add_argument('--loss_weights', type=str, default="", help='loss weights for different class')
    parser.add_argument('--model_name', type=str, default="resnet18", help='resnet18 or resnet50')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--num_epoch', type=int, default=30, help='number of epoch for training.')
    parser.add_argument('--feature_extract', type=bool, default=False, help='whether training the feature extractor')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help='whether to use pretrained weights')
    parser.add_argument('--log_every_n_epoch', type=int, default=1, help='log model every n epoch')
    parser.add_argument('--polyaxon', action='store_true', help='polyaxon', default=False)
    parser.add_argument('--eval_only', action='store_true', help='whether eval only', default=False)
    parser.add_argument('--resume', action='store_true', help='whether resume from last checkpoint', default=False)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0005)
    parser.add_argument('--eval_phase', type=str, help='evalation phase', default="val")

    args = parser.parse_args()
    launch(
        main,
        num_gpus_per_machine=args.num_gpu,
        dist_url="tcp://127.0.0.1:28662",
        num_machines=1,
        machine_rank=0,
        args=(args,))
