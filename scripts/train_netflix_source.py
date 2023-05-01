"""
Train Netflix Source Models

Usage:
    train_netflix_source.py [--target-size=<ts>] [--dataset-foldername=<dfn>] [--single-threaded=<st>]

Options:
    -h --help                   Shows this screen.
    --target-size=<ts>          Defines the size of the target dataset, also defining the size of the source. [default: 100k]
    --dataset-foldername=<dfn>  Name of the dataset folder. [default: ml100k]
    --single-threaded=<st>      Runs this code using a single CPU core. [default: False]
"""

from docopt import docopt
import sys
from typing import List
sys.path.append('../')

from src.config import set_singlethreaded
from src.training_util import train_netflix_source
from src.util import (
    create_netflix_source_folder,
    get_available_netflix_source_configs,
    initialize_logging,
    parse_netflix_arguments,
    split_configs_in_queues,
)


def train_models(arguments: dict, configs: List) -> None:
    source_kwargs = parse_netflix_arguments(arguments)

    queues = split_configs_in_queues(configs)
    train_netflix_source(queues, source_kwargs)


if __name__ == '__main__':
    initialize_logging()
    arguments = docopt(__doc__, version='Train Netflix Source Models v0.0.1')
    set_singlethreaded(arguments)
    create_netflix_source_folder(arguments)
    configs = get_available_netflix_source_configs(arguments)

    train_models(arguments, configs)
