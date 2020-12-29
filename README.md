## Introduction
dlengine is a trainning engine targeting to separating miscellaneous trainning things from modeling.


With dlengine, you can focuse on building model. dlengine takes care of logging, model saving and restoring, very efficient multi-gpu training for you.


dlengine contains features:
- very efficient multi-gpu training based on `DistributedDataParallel`
- automatic logging 
- automatic model saving and restoring


## Usage
step 1. inheriting from `DefaultTrainer`，overwrite neccesary methods `build_model`, `build_optimizer`, `build_lr_scheduler`, `build_train_loader`, `build_test_loader`, `build_evaluator`.
```python
from dlengine.defaults import DefaultTrainer

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

step 2. call `lanuch`function to start training

```python
from dlengine.launch import launch
from dlengine.defaults import default_setup
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

## Installation

1. install package dependencies: pytorch and termcolor. 

2. `pip install dlengine`


## LICENSE

Copyright (c) 2021 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.