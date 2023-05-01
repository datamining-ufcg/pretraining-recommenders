import sys
sys.path.append('../')

import src.io_util as io
from src.catalog_util import *


catalog = {}
summary = {}


def catalog_folder(path: str) -> None:
    """
    Catalogs all models inside the given folder,
    updating the global catalog.
    """
    global catalog, summary

    folder_names = io.listdir(path)
    summary[path] = len(folder_names)

    for model_folder in folder_names:
        add_to_catalog(path, model_folder, catalog)


def recursive_catalog(path: str):
    for folder_name in io.listdir(path, filter_files=True):
        current_path = f'{path}/{folder_name}'
        if (folder_name == 'source' or path.endswith('target')):
            catalog_folder(current_path)

        else:
            recursive_catalog(current_path)


MODELS_FOLDER = '../models'
recursive_catalog(MODELS_FOLDER)

kwargs = {'indent': 4, 'sort_keys': True}

io.save_json('../models', 'catalog', catalog, **kwargs)
io.save_json('../models', 'summary', summary, **kwargs)
print(io.show_json(summary, **kwargs))
