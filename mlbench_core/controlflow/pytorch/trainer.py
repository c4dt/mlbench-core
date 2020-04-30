import logging

import torch
import torch.optim
import torch.utils.data
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average
from mlbench_core.utils.pytorch.helpers import iterate_dataloader

logger = logging.getLogger("mlbench")
LOG_EVERY_N_BATCHES = 25


def _record_train_batch_stats(
    batch_idx, loss, output, target, metrics, tracker, num_batches_per_device_train
):
    r"""Record the stats in a training batch.

    Args:
        batch_idx (int): The id of the current batch
        loss (float): The loss of the batch
        output (:obj:`torch.Tensor`): The model output
        target (:obj:`torch.Tensor`): The labels for the current batch
        metrics (list): List of metrics to track
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use.
        num_batches_per_device_train (int): Number of batches per train epoch
    """
    progress = batch_idx / num_batches_per_device_train
    progress += tracker.current_epoch

    log_to_api = (
        batch_idx % LOG_EVERY_N_BATCHES == 0
        or batch_idx == num_batches_per_device_train
    )

    if tracker:
        tracker.record_loss(loss, output.size()[0], log_to_api=log_to_api)

    # Compute metrics for one batch
    for metric in metrics:
        metric_value = metric(loss, output, target).item()

        if tracker:
            tracker.record_metric(
                metric, metric_value, output.size()[0], log_to_api=log_to_api
            )

    status = "Epoch {:5.2f} Batch {:4}: ".format(progress, batch_idx)

    logger.info(status + str(tracker))


class MLBenchTrainer:
    """Trainer class for pytorch models. This class can be seen as a super class that implements basic methods
    that would work for most models. However, some new models might require the re-definition of some functions
    such as `compute_model_output` and `compute_loss`, that could be specific to certain models. (e.g. GNMT)

    Args:
        model (`obj`:torch.nn.Module): Model to train
        optimizer (`obj`:torch.optim): The optimizer
        loss_function (`obj`:torch.nn.Module): The loss function
        scheduler (`obj`:torch.optim.lr_scheduler): Learning Rate scheduler
        schedule_per (str): Scheduler per `epoch` or `batch`
        metrics (list): List of metrics to track
        use_cuda (bool): Whether to use GPU for training. Default: `False`
        rank (int): The rank of the current worker node. Default: 0
        tracker (`obj`:mlbench_core.utils.Tracker): Tracker object to use. Default `None`
        empty_cache (bool): Empty cache before every training epoch and validation round.
            Default: `False`
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        scheduler,
        metrics,
        schedule_per="epoch",
        use_cuda=False,
        rank=0,
        tracker=None,
        empty_cache=False,
    ):
        assert model is not None, "Please provide a model"
        assert optimizer is not None, "Please provide an optimizer"
        assert loss_function is not None, "Please provide a loss function"
        assert scheduler is not None, "Please provide a learning rate scheduler"
        assert (
            schedule_per == "epoch" or schedule_per == "batch"
        ), "Scheduler per should be (epoch | batch)"
        assert metrics is not None, "Please provide some metrics"

        self.model = model
        self.rank = rank
        self.tracker = tracker
        self.empty_cache = empty_cache
        self.schedule_per = schedule_per
        self.scheduler = scheduler
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.loss_function = loss_function

    def _training(self):
        """Sets the model and tracker in training"""
        self.model.train()

        if self.tracker:
            self.tracker.train()

    def _eval(self):
        """Sets the model and tracker in evaluation mode"""
        self.model.eval()

        # Set tracker in validation mode
        if self.tracker:
            self.tracker.validation()
            self.tracker.validation_start()

    def optimize(self, batch_idx, data, target, num_batches_per_device_train):
        """Performs one optimization step

        Args:
            batch_idx (int): Index of the current batch
            data (`obj`:torch.Tensor): Input tensor
            target (`obj`:torch.Tensor): Target tensor
            num_batches_per_device_train (int): Number of batches per device per epoch
        """
        tracker = self.tracker
        if tracker:
            tracker.batch_start()

        # Clear gradients in the optimizer.
        self.optimizer.zero_grad()
        if tracker:
            tracker.record_batch_step("init")

        # Compute the output
        output = self.compute_model_output(data, target)
        if tracker:
            tracker.record_batch_step("fwd_pass")

        # Compute the loss
        loss = self.compute_loss(output, target, output)
        if tracker:
            tracker.record_batch_step("comp_loss")

        # Backprop
        loss.backward()
        if tracker:
            tracker.record_batch_step("backprop")

        # Aggregate gradients/parameters from all workers and apply updates to model
        self.optimizer.step()
        if tracker:
            tracker.record_batch_step("opt_step")

        if self.schedule_per == "batch":
            self.scheduler.step()

        if tracker:
            tracker.batch_end()

        _record_train_batch_stats(
            batch_idx,
            loss.item(),
            output,
            target,
            self.metrics,
            tracker,
            num_batches_per_device_train,
        )

    def compute_loss(self, data, target, output):
        """Computes the loss

        Args:
            data (`obj`:torch.Tensor): Data input
            target (`obj`:torch.Tensor): Target
            output (`obj`:torch.Tensor): Model output

        Returns:
            (`obj`:torch.Tensor) The result of the loss function
        """
        return self.loss_function(output, target)

    def compute_model_output(self, data, target):
        """Computes the output of the model

        Args:
            data (`obj`:torch.Tensor): Input tensor
            target (`obj`:torch.Tensor): Target tensor

        Returns:
            (`obj`:torch.Tensor): The model's output
        """
        return self.model(data)

    def _iterate_loader(
        self, dataloader, dtype, max_batch_per_epoch, transform_target_type
    ):
        """Returns an iterator on the data loader

        Args:
            dataloader (`obj`:torch.utils.data.DataLoader): The loader to iterate onto
            dtype (str): The datatype to use, one of `fp32`or `fp64`
            max_batch_per_epoch (int): Maximum number of batches tot rain for per epoch
            transform_target_type (bool): Transform the target type too

        Returns:
            iterator: An iterator on the given loader
        """
        return iterate_dataloader(
            dataloader, dtype, max_batch_per_epoch, self.use_cuda, transform_target_type
        )

    def _update_val_losses(self, data, target, output, losses):
        """Computes the loss for the given output and updates the AverageMeter

        Args:
            data (`obj`:torch.Tensor): Input tensor
            target (`obj`:torch.Tensor): Target tensor
            output (`obj`:torch.Tensor): Output tensor
            losses (AverageMeter): The running average

        Returns:
            (`obj`:torch.Tensor): The loss of the given data point
        """
        # Compute loss
        loss = self.compute_loss(output, target, output)

        # Update loss
        losses.update(loss.item(), data.size(0))

        return loss

    def _update_val_metrics(self, data, target, output, loss):
        """Updates and computes the metrics for the given data point

        Args:
            data (`obj`:torch.Tensor): Input tensor
            target (`obj`:torch.Tensor): Target tensor
            output (`obj`:torch.Tensor): Output tensor
            loss (`obj`:torch.Tensor): Loss tensor
        """
        # Update metrics
        for metric in self.metrics:
            metric_value = metric(loss, output, target)
            metric.update(metric_value, data.size(0))

    def train_round(
        self,
        train_loader,
        dtype,
        max_batch_per_epoch=None,
        transform_target_type=False,
        val_loader=None,
        validate_every=None,
    ):
        """ Performs one epoch of training

        Args:
            train_loader (`obj`:torch.utils.data.DataLoader): The train set loader
            dtype (str): The datatype to use for iterator, one of `fp32`or `fp64`
            max_batch_per_epoch (int | None): Maximum number of batches tot rain for per epoch,
                Default: `None` (all batches)
            transform_target_type (bool): Transform the target type too.
                Default: `False`
            val_loader (`obj`:torch.utils.data.DataLoader | None): The validation set loader.
                Default: `None` (no validation during training)
            validate_every (int | None): Validate every n batches.
                Default: `None` (no validation)
        """
        # Set in training mode
        self._training()

        validate = validate_every is not None and val_loader is not None
        num_batches_per_device_train = len(train_loader)

        # Empty cache
        if torch.cuda.is_available() and self.empty_cache:
            torch.cuda.empty_cache()

        # Get data iterator
        data_iter = self._iterate_loader(
            train_loader, dtype, max_batch_per_epoch, transform_target_type
        )

        # Start epoch
        for batch_idx, (data, target) in enumerate(data_iter):
            self.optimize(batch_idx, data, target, num_batches_per_device_train)
            if validate and (batch_idx + 1) % validate_every == 0:
                self.validation_round(
                    val_loader, dtype, max_batch_per_epoch, transform_target_type
                )
                self._training()

        if self.schedule_per == "epoch":
            self.scheduler.step()

    def validate(
        self, val_loader, dtype, max_batch_per_epoch=None, transform_target_type=False
    ):
        """Performs validation of the validation set

        Args:
            val_loader (`obj`:torch.utils.data.DataLoader): The data set loader
            dtype (str): The datatype to use for iterator, one of `fp32`or `fp64`
            max_batch_per_epoch (int): Maximum number of batches tot rain for per epoch,
                Default: `None` (all batches)
            transform_target_type (bool): Transform the target type too.
                Default: `False`

        Returns:
            (dict, float): The metrics averages and the loss average
        """
        losses = AverageMeter()

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            data_iter = self._iterate_loader(
                val_loader, dtype, max_batch_per_epoch, transform_target_type
            )
            for data, target in data_iter:
                # Compute output
                output = self.compute_model_output(data, target)

                # Compute and update losses
                loss = self._update_val_losses(data, target, output, losses)

                # Compute and update metrics
                self._update_val_metrics(data, target, output, loss)

        metrics_averages = {metric: metric.average().item() for metric in self.metrics}
        loss_average = global_average(losses.sum, losses.count).item()
        return metrics_averages, loss_average

    def validation_round(
        self, val_loader, dtype, max_batch_per_epoch=None, transform_target_type=False
    ):
        """Performs one validation round and checks if the goal was reached

        Args:
            val_loader (`obj`:torch.utils.data.DataLoader): The validation set
            dtype (str): The datatype to use for iterator, one of `fp32`or `fp64`
            max_batch_per_epoch (int | None): Maximum number of batches tot rain for per epoch,
                Default: `None` (all batches)
            transform_target_type (bool): Transform the target type too.
                Default: `False`
        Returns:
            (bool): Whether this validation is the best so far
        """
        if torch.cuda.is_available() and self.empty_cache:
            torch.cuda.empty_cache()

        tracker = self.tracker
        self._eval()
        # Gather metrics and loss average
        metrics_values, loss = self.validate(
            val_loader, dtype, max_batch_per_epoch, transform_target_type
        )
        if tracker:
            tracker.validation_end()

        if len(metrics_values) > 0:
            # Save metrics
            if tracker:
                for metric, value in metrics_values.items():
                    tracker.record_metric(metric, value, log_to_api=True)

                    global_metric_value = global_average(value, 1).item()

                    if self.rank == 0:
                        tracker.record_stat(
                            "global_{}".format(metric.name),
                            global_metric_value,
                            log_to_api=True,
                        )

            # Update logger at rank 0
            if self.rank == 0 and tracker:
                logger.info(
                    "{} for rank {}:(best epoch {}, current epoch {}): {:.3f}".format(
                        tracker.primary_metric.name,
                        tracker.rank,
                        tracker.best_epoch,
                        tracker.current_epoch,
                        tracker.best_metric_value,
                    )
                )
        else:
            if self.rank == 0:
                logger.info("Validation loss={:.3f}".format(loss))

        if tracker:
            tracker.record_loss(loss, log_to_api=True)

            global_loss = global_average(loss, 1).item()

            if self.rank == 0:
                tracker.record_stat("global_loss", global_loss, log_to_api=True)

        return tracker.is_best() if tracker else False
