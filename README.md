
# Experiment Logger

TensorboardX and lightweight logging for machine learning experiments in PyTorch.
I find it useful to have the ability to view plots in Tensorboard and to constuct
nicer plots and CSVs from the raw log files.
 
The lightweight logger is adapted from James Lucas. 

## Usage

Initialize by pointing the logger at a directory.

```python
from exp_utils import ExperimentLogger
logger = ExperimentLogger(log_dir)
```

At the end of training loop (or every `k` iterations):
```python
epoch_stats = {"train_acc": train_acc1,
               "test_acc": val_acc1,
               "test_loss": val_loss, }
state = {
    'epoch': epoch + 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
logger.log_epoch(epoch, state, epoch_stats)
```

To resume an experiment:

```python
model = # define model
optimizer = # define optimizer
if logger.checkpoint_exists():
        checkpoint = logger.load_checkpoint()
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("Resuming training from epoch {}".format(start_epoch))
        # set up decay
        for i in range(start_epoch):
            scheduler.step()

        logger.update_tensorboard_writer(start_epoch)
```