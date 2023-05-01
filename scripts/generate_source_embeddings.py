"""
Generate Source Embeddings

Usage:
    generate_source_embeddings.py --target=<t>

Options:
    --target=<t>    Defines which dataset size is the target for the transference. [default: ml1m]
    
"""
from docopt import docopt
import itertools
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('../')
from src.config import RANDOM_STATE, EMBEDDING_SIZE, REG
from src.dataset.ml_transfer import ExplicitMLTransfer
from src.models.CythonSVD import CythonSVD
from src.util import (
    build_path_from_source_kwargs,
    create_source_folder,
    parse_source_arguments,
)

SEED = RANDOM_STATE
np.random.seed(SEED)


def combine_params(target):
    dataset_sizes = ['1m', '10m', '20m', '25m']
    params = list(itertools.product(
        dataset_sizes,
        [target],
        zip(
            ['random', 'pca', 'word2vec'],
            [{'stddev': 0.1}, {}, {'window_size': 5}]
        ),
    ))

    return params


def create_arguments(target, dataset):
    args = {
        '--target-size': target,
        '--dataset-foldername': f'leakage_user_from_{dataset}',
        '--items-filename': '',
        '--items-separator': '',
    }

    return args


def generate_embeddings(params):
    for dataset, target, (inst, inpr) in tqdm(params):
        if (target == '1m' and dataset == '1m'):
            continue

        print(dataset, target, inst)
        args = create_arguments(target, dataset)
        create_source_folder(args)
        kwargs = parse_source_arguments(args)
        output_folder = build_path_from_source_kwargs(kwargs)

        dataset = ExplicitMLTransfer(**kwargs)
        dataset.load()
        dataset.get_fold(k=1)

        _ = CythonSVD(
            dataset, EMBEDDING_SIZE, inst, inpr, REG,
            model_path=output_folder, skip_folder_creation=True,
        )


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Generate Source Model Embeddings v0.0.1')
    target = arguments['--target']

    params = combine_params()
    generate_embeddings(params)
