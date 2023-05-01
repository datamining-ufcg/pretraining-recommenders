"""
Train Target Models

Usage:
    train_target_model.py --target-size=<ts> --dataset-foldername=<dfn> [--items-filename=<ifn>] [--items-separator=<isp>]

Options:
    -h --help                   Shows this screen.
    --target-size=<ts>          Defines the size of the target dataset, also defining the size of the source.
    --dataset-foldername=<dfn>  Name of the dataset folder.
    --items-filename=<ifn>      Name of the items file [default: movies.csv]
    --items-separator=<isp>     Characters for the item separator [default: ,]
"""

from docopt import docopt
import logging
import sys
from typing import List, Any

sys.path.append('../')


from src.dataset.ml_transfer import ExplicitMLTransfer
from src.exception import UnexistingSourceException
from src.models.CythonSVD import CythonSVD
from src.training_util import train_svd_target
from src.util import (
    create_transfer_folders,
    get_available_target_configs,
    get_source_model_path,
    initialize_logging,
    parse_transfer_arguments,
    split_configs_in_queues,
)


def load_source(source_kwargs: dict, source_method: str) -> Any:
    source_model_path = get_source_model_path(source_kwargs, source_method)
    source_model = CythonSVD.load_model(source_model_path)

    source_dataset = ExplicitMLTransfer(**source_kwargs)
    source_dataset.load_item_description()
    source_dataset.load_inner_mappings()

    return source_model, source_dataset


def train_models(arguments: dict, configs: List) -> None:
    source_kwargs, target_kwargs = parse_transfer_arguments(arguments)

    for available, source_method in configs:
        try:
            source_model, source_dataset = load_source(
                source_kwargs, source_method
            )
        except UnexistingSourceException:
            logging.debug('Source not found. Skipping.')
            continue

        num_halted = len([1 for _, path in available if path != ''])

        logging.debug(
            f'Source is loaded. {len(available)} configs will be trained, {num_halted} of them being halted.')
        queues = split_configs_in_queues(available)
        train_svd_target(queues, source_model, source_dataset, target_kwargs)

    if (len(configs) == 0):
        logging.debug('All the models were fully trained.')


if __name__ == '__main__':
    initialize_logging()
    arguments = docopt(__doc__, version='Train Target Models v0.0.2')
    create_transfer_folders(arguments, use_any=True)
    configs = get_available_target_configs(arguments)

    train_models(arguments, configs)
