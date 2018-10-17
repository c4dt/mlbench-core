import time
import datetime
import itertools
import shutil
import os
import logging
import socket
import random
import torch
import torch.distributed as dist

from mlbench_core.utils.pytorch import checkpoint
from mlbench_core.utils.pytorch.topology import FCGraph


class Timeit(object):
    def __init__(self, cumu=0):
        self.t = time.time()
        self._cumu = cumu
        self._paused = False

    def pause(self):
        if not self._paused:
            self._cumu += time.time() - self.t
            self.t = time.time()
            self._paused = True

    def resume(self):
        if self._paused:
            self.t = time.time()
            self._paused = False

    @property
    def cumu(self):
        return self._cumu


def maybe_range(maximum):
    """Map an integer or None to an integer iterator starting from 0 with strid 1.

    If maximum number of batches per epoch is limited, then return an finite
    iterator. Otherwise, return an iterator of infinite length. 
    """
    if maximum is None:
        counter = itertools.count(0)
    else:
        counter = range(maximum)
    return counter


def update_best_runtime_metric(config, metric_value, metric_name):
    """Update the runtime information to config if the metric value is the best."""
    best_metric_name = "best_{}".format(metric_name)
    if best_metric_name in config.runtime:
        is_best = metric_value > config.runtime[best_metric_name]
    else:
        is_best = True

    if is_best:
        config.runtime[best_metric_name] = metric_value
        config.runtime['best_epoch'] = config.runtime['current_epoch']

    return is_best, best_metric_name


def convert_dtype(dtype, obj):
    # The object should be a ``module`` or a ``tensor``
    if dtype == 'fp32':
        return obj.float()
    elif dtype == 'fp64':
        return obj.double()
    else:
        raise NotImplementedError('dtype {} not supported.'.format(dtype))


def config_logging(config):
    """Setup logging modules.

    A stream handler and file handler are added to default logger `mlbench`.
    """

    level = config.logging_level
    logging_file = config.logging_file

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = dist.get_rank()
            return True

    logger = logging.getLogger('mlbench')
    if len(logger.handlers) >= 2:
        return

    logger.setLevel(level)
    logger.addFilter(RankFilter())

    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(rank)s %(levelname)s: %(message)s',
        "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logging_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def config_pytorch(config):
    """Config pytorch packages.

    Fix random number for packages and initialize distributed environment for pytorch.
    Setup cuda environment for pytorch.

    :param config: A global object containing specified config.
    :type config: argparse.Namespace
    """
    # Setting `cudnn.deterministic = True` will turn on
    # CUDNN deterministic setting which can slow down training considerably.
    # Unexpected behavior may also be observed from checkpoint.
    # See: https: // github.com/pytorch/examples/blob/master/imagenet/main.py
    if config.cudnn_deterministic:
        cudnn.deterministic = True
        print('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    # define the graph for the computation.
    if config.use_cuda:
        assert torch.cuda.is_available()

    config.rank = dist.get_rank()
    config.world_size = dist.get_world_size()
    config.graph = FCGraph(config)

    # enable cudnn accelerator if we are using cuda.
    if config.use_cuda:
        config.graph.assigned_gpu_id()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if torch.backends.cudnn.version() is None:
            print("CUDNN not found on device.")

        print("World size={}, Rank={}, hostname={}, cuda_available={}, cuda_device={}".format(
            config.world_size, config.rank, socket.gethostname(), torch.cuda.is_available(),
            torch.cuda.current_device()))


def log_metrics(config, metric_name, value):
    data = {
        "run_id": config.run_id,
        "rank": config.rank,
        "name": metric_name,
        "value": "{:.6f}".format(value),
        "date": str(datetime.datetime.now()),
        "epoch": str(config.runtime['current_epoch']),
        "cumulative": "False",
        "metadata": ""
    }
    config.runtime['records'].append(data)


def config_path(config):
    """Config the path used during the experiments."""

    # Checkpoint for the current run
    config.ckpt_run_dir = checkpoint.get_ckpt_run_dir(
        config.checkpoint_root, config.run_id,
        config.dataset, config.model, config.optim)

    if not config.resume:
        print("Remove previous checkpoint directory : {}".format(
            config.ckpt_run_dir))
        shutil.rmtree(config.ckpt_run_dir, ignore_errors=True)
    os.makedirs(config.ckpt_run_dir, exist_ok=True)


def initialize(config):
    if not (hasattr(dist, '_initialized') and dist._initialized):
        dist.init_process_group(config.comm_backend)

    config_logging(config)

    config_pytorch(config)

    config_path(config)

    return config
