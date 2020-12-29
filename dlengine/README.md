### 介绍
本模型训练引擎是将与模型训练相关的日志输出、模型保存、模型加载、高效多GPU训练等与模型训练相关的任务，独立形成模块，让科学家更专注于模型。

本模块，包含了：
- 基于`DistributedDataParallel`的高效多GPU训练
- 日志自动输出与保存
- 模型自动保存与加载

本模块，具有以下特点：
- 日志输出友好
- 扩展性好
- 非常容易集成进现有项目中
- 使用本模型后，代码逻辑更清晰

### 项目集成
step 1. 继承类`DefaultTrainer`，重写 `build_model`、`build_optimizer`、`build_lr_scheduler`、`build_train_loader`、`build_test_loader`、`build_evaluator`等方法。
```
from classifier.engine.defaults import DefaultTrainer

class Trainer(DefaultTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args, args.output_dir, "classifier", args.num_epoch, args.log_every_n_epoch)

    @classmethod
    def build_model(cls, args):
        pass

    @classmethod
    def build_optimizer(cls, args, model):
        pass

    @classmethod
    def build_lr_scheduler(cls, args, end_iter, optimizer):
        pass

    @classmethod
    def build_train_loader(cls, args):
        pass

    @classmethod
    def build_test_loader(cls, args):
        pass

    @classmethod
    def build_evaluator(cls, args):
        pass
```

step 2. 调用`lanuch`函数，启动训练

```
from classifier.engine.launch import launch
from classifier.engine.defaults import default_setup
def main(args):
    default_setup(args.output_dir, "classifier", args)
    trainer = Trainer(args)
    return trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    launch(
        main,
        num_gpus_per_machine=args.num_gpu,
        dist_url="tcp://127.0.0.1:28662",
        num_machines=1,
        machine_rank=0,
        args=(args,))
```