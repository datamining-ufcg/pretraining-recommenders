"""
Train Source Models

Usage:
    train_source_model.py --target-size=<ts> --dataset-foldername=<dfn> [--items-filename=<ifn>] [--items-separator=<isp>]

Options:
    -h --help                   Shows this screen.
    --target-size=<ts>          Defines the size of the target dataset, also defining the size of the source.
    --dataset-foldername=<dfn>  Name of the dataset folder.
    --items-filename=<ifn>      Name of the items file [default: movies.csv]
    --items-separator=<isp>     Characters for the item separator [default: ,]
"""

from docopt import docopt
import sys
from typing import List
sys.path.append('../')


from src.training_util import train_svd_source
from src.util import (
    create_source_folder,
    initialize_logging,
    get_available_source_configs,
    parse_source_arguments,
    split_configs_in_queues,
)


def train_models(arguments: dict, configs: List) -> None:
    source_kwargs = parse_source_arguments(arguments)

    queues = split_configs_in_queues(configs)
    train_svd_source(queues, source_kwargs)


if __name__ == '__main__':
    initialize_logging()
    arguments = docopt(__doc__, version='Train Source Models v0.0.2')
    create_source_folder(arguments)
    configs = get_available_source_configs(arguments)

    train_models(arguments, configs)
