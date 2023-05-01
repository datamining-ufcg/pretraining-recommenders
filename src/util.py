from typing import List, Tuple

from src.config import LOGGING_FORMAT
from src.catalog_util import check_source_on_catalog, filter_target_params_with_catalog
from src.exception import UnexistingSourceException
from src.io_util import create_folder, load_json, listdir


def combine_source_params(splits: bool = False):
    """
    Generates the combination of parameters. The parameters are the method of initialization.
    """
    import itertools

    splits = range(1, 6) if splits else [1]

    params = list(
        itertools.product(
            zip(
                ["random", "pca", "word2vec"], [{"stddev": 0.1}, {}, {"window_size": 5}]
            ),
            splits,
        )
    )

    return params


def combine_target_params():
    """
    Generates the combination of parameters. The parameters are the method of initialization,
    the percentage of the dataset allowed to transfer, and the dataset split used according
    the k-fold cross-validation.
    """
    import itertools

    params = list(
        itertools.product(
            zip(
                ["random", "pca", "word2vec"], [{"stddev": 0.1}, {}, {"window_size": 5}]
            ),
            range(11),
            range(1, 6),
        )
    )

    return params


def build_path_from_source_kwargs(source_kwargs: dict) -> str:
    """
    Builds
    """
    target_folder = "target_" + source_kwargs["target_size"]
    dataset_foldername = source_kwargs["dataset_foldername"].split("_")
    dataset_foldername = "_".join([ti for ti in dataset_foldername if ti != "from"])
    dataset_foldername = dataset_foldername.replace("_negative", "")

    path = f"../models/{target_folder}/{dataset_foldername}/source"
    return path


def build_path_from_target_kwargs(
    target_kwargs: dict, source_method: str,
) -> str:
    target_folder = "target_" + target_kwargs["target_size"]
    dataset_foldername = target_kwargs["dataset_foldername"].split("_")
    dataset_foldername = "_".join([ti for ti in dataset_foldername if ti != "from"])

    path = f"../models/{target_folder}/{dataset_foldername}/target"
    start = (
        "leakage_" if target_kwargs["dataset_foldername"].startswith("leakage") else ""
    )
    path += f'/{start}transfer_{target_kwargs["target_size"]}_{source_method}'

    return path


def get_source_model_path(
    source_kwargs: dict, source_method: str, split: int = 1
) -> str:
    """
    Retrieves source model trained with the given initialization method.
    If no model is found, raises an exception.
    """

    path = build_path_from_source_kwargs(source_kwargs)
    for folder in listdir(path):
        folder_path = f"{path}/{folder}"
        config = load_json(folder_path, "config")
        if config["method"] == source_method and config["dataset_split"] == split:
            return folder_path

    raise UnexistingSourceException()


def get_target_model_path(qp: Tuple[Tuple[str, dict], int, int], path: str) -> str:
    (inst, _), limit, split = qp
    for folder in listdir(path):
        folder_path = f"{path}/{folder}"
        config = load_json(folder_path, "config")
        if (
            config["init"] == inst
            and config["limit"] == (limit / 10)
            and config["dataset_split"] == split
        ):
            return folder_path

    raise UnexistingSourceException()


def parse_source_arguments(args: dict) -> dict:
    source_kwargs = {
        "target_size": args["--target-size"],
        "dataset_foldername": args["--dataset-foldername"] + "_negative",
        "ratings_filename": "negative_ml" + args["--target-size"],
        "items_filename": args["--items-filename"],
        "items_separator": args["--items-separator"],
        "is_source": True,
    }

    return source_kwargs


def parse_transfer_arguments(args: dict) -> Tuple[dict, dict]:
    target_kwargs = {
        "target_size": args["--target-size"],
        "dataset_foldername": args["--dataset-foldername"],
        "ratings_filename": "ml" + args["--target-size"],
        "items_filename": args["--items-filename"],
        "items_separator": args["--items-separator"],
        "is_source": False,
    }
    source_kwargs = parse_source_arguments(args)

    return source_kwargs, target_kwargs


def _parse_arguments(arguments):
    """
    Parses arguments to obtain target folder and dataset foldername
    """
    target_folder = "target_" + arguments["--target-size"]
    dataset_foldername = arguments["--dataset-foldername"].split("_")
    dataset_foldername = "_".join([ti for ti in dataset_foldername if ti != "from"])

    return target_folder, dataset_foldername


def create_source_folder(arguments):
    target_folder, dataset_foldername = _parse_arguments(arguments)
    path = "../models"
    create_folder(path, target_folder)

    path += f"/{target_folder}"
    ds_foldername = dataset_foldername.replace("_negative", "")
    create_folder(path, ds_foldername)

    path += f"/{ds_foldername}"
    create_folder(path, "/source")


def create_transfer_folders(arguments, use_any=False):
    """
    Create the necessary folders to allow transferring items
    from the already trained source models to the target ones.
    """

    target_folder, dataset_foldername = _parse_arguments(arguments)
    methods = ["random", "pca", "word2vec"]
    sources = [
        check_source_on_catalog(target_folder, dataset_foldername, v)[0]
        for v in methods
    ]

    if (use_any and not any(sources)) or (not use_any and not all(sources)):
        raise UnexistingSourceException()

    path = f"../models/{target_folder}/{dataset_foldername}"
    path = path.replace("_negative", "")
    create_folder(path, "target")
    path += "/target"

    method_folder = "leakage_transfer_{}_{}"
    for m in methods:
        create_folder(path, method_folder.format(arguments["--target-size"], m))


def get_available_source_configs(
    arguments: dict, every: bool = False
) -> List[Tuple[bool, str]]:
    target_folder, dataset_foldername = _parse_arguments(arguments)

    all_configs = combine_source_params(every)
    available = []

    for (init_method, init_method_params), split in all_configs:
        filtered = check_source_on_catalog(
            target_folder, dataset_foldername, init_method, split
        )
        if not filtered[0]:
            available.append(((init_method, init_method_params, split), filtered[1]))

    return available


def get_available_target_configs(arguments):
    """
    Retrieves a list of available configurations for training that either
    the training hasn't started or didn't conclude.
    """
    target_folder, dataset_foldername = _parse_arguments(arguments)
    all_configs = combine_target_params()
    available_configs = []

    for parent_method in ["pca", "random", "word2vec"]:
        filtered = filter_target_params_with_catalog(
            target_folder, dataset_foldername, parent_method, all_configs
        )
        if len(filtered) == 0:
            continue

        available_configs.append((filtered, parent_method))

    return available_configs


def split_configs_in_queues(configs):
    from src.config import MAX_THREADS

    queues = [[] for _ in range(MAX_THREADS)]
    for idx, p in enumerate(configs):
        qidx = idx % MAX_THREADS
        queues[qidx].append(p)

    return queues


def initialize_logging():
    import logging

    logging.basicConfig(
        level=logging.DEBUG, format=LOGGING_FORMAT,
    )


def parse_netflix_arguments(args: dict) -> dict:
    """
    Receives the inline arguments from docopt and parses them to generate
    the dict needed to initialize a NetflixTransfer dataset.
    """
    ts = args["--target-size"]
    ts = ts.upper() if ts == "1m" else ts
    ratings_filename = f"ExplicitML{ts}_netflix_mapped.csv"

    netflix_source_kwargs = {
        "target_size": args["--target-size"],
        "dataset_foldername": args["--dataset-foldername"],
        "ratings_filename": ratings_filename,
        "is_source": True,
    }

    return netflix_source_kwargs


def parse_netflix_transfer_arguments(args: dict) -> dict:
    source_kwargs = parse_netflix_arguments(args)
    target_kwargs = {
        "target_size": args["--target-size"],
        "dataset_foldername": args["--dataset-foldername"],
        "name": args["--dataset-foldername"],
    }

    return source_kwargs, target_kwargs


def create_netflix_source_folder(arguments: dict) -> None:
    """
    Creates the necessary folders to organize pre-training and transfer
    from a netflix dataset to a movielens target.
    """

    target_folder, dataset_foldername = _parse_arguments(arguments)
    path = "../models"
    create_folder(path, target_folder)

    path += f"/{target_folder}"
    create_folder(path, dataset_foldername)

    path += f"/{dataset_foldername}"
    create_folder(path, "/source")


def create_netflix_transfer_folders(arguments, use_any=False):
    target_folder, dataset_foldername = _parse_arguments(arguments)
    methods = ["random", "pca", "word2vec"]
    sources = [
        check_source_on_catalog(target_folder, dataset_foldername, v)[0]
        for v in methods
    ]

    if (use_any and not any(sources)) or (not use_any and not all(sources)):
        raise UnexistingSourceException()

    path = f"../models/{target_folder}/{dataset_foldername}"
    create_folder(path, "target")
    path += "/target"

    method_folder = "transfer_{}_{}"
    for m in methods:
        create_folder(path, method_folder.format(arguments["--target-size"], m))


def combine_netflix_source_params() -> List[Tuple[str, dict]]:
    return list(
        zip(["random", "pca", "word2vec"], [{"stddev": 0.1}, {}, {"window_size": 5}])
    )


def get_available_netflix_source_configs(arguments: dict) -> List[Tuple[bool, str]]:
    """
    """
    target_folder, dataset_foldername = _parse_arguments(arguments)

    all_configs = combine_netflix_source_params()
    available = []

    for init_method, init_method_params in all_configs:
        filtered = check_source_on_catalog(
            target_folder, dataset_foldername, init_method
        )
        if not filtered[0]:
            available.append(((init_method, init_method_params), filtered[1]))

    return available
