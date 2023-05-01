# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
cimport numpy as np  # noqa
from datetime import datetime
from abc import ABC, abstractmethod, abstractstaticmethod
from collections import defaultdict

import numpy as np
import src.io_util as io
from src.exception import UndefinedOutputException


class CythonModel(ABC):
    """
    A superclass for model implementation with Cython.
    """

    def __init__(
        self,
        dataset,
        embedding_dim,
        method,
        method_args,
        reg,
        cold_start=True,
        model_path='',
        skip_folder_creation=False
    ):
        super(CythonModel, self).__init__()

        self.method = method
        self.method_args = method_args
        self.n_factors = embedding_dim
        self.reg = reg

        if (cold_start):
            self._init_dataset_attributes(dataset)

        self._init_attributes()
        if (not skip_folder_creation):
            self._create_model_folder(model_path)
        elif (model_path != ''):
            self.model_path = model_path
        else:
            raise UndefinedOutputException()

    def _init_dataset_attributes(self, dataset):
        self.pu = dataset.get_embeddings(
            self.n_factors, 'random', user=True, **{'stddev': 0.1}
        )
        self.qi = dataset.get_embeddings(
            self.n_factors, self.method, **self.method_args
        )
        self.global_bias = dataset.trainset.global_mean
        self.num_users = dataset.trainset.n_users
        self.num_items = dataset.trainset.n_items
        self.num_ratings = dataset.trainset.n_ratings
        self.dataset_name = dataset.name
        self.dataset_split = dataset.current_fold
        self.dataset_sparsity = dataset.sparsity()

    def _init_attributes(self):
        self.results = defaultdict(list)
        self.n_emb = 0
        self.limit = 1
        self.init_strategy = None
        self.init_params = None

    @abstractmethod
    def predict(self, u, i):
        pass

    @abstractmethod
    def fit(self, trainset, testset, epochs, learning_rate, patience, verbose=True):
        pass

    def use_pretrained_items(
        self,
        pretrained_embeddings,
        item_mappings,
        ds_name,
        ds_method,
        limit=1
    ):
        """
        Use previously generated embeddings to initialize the current model.
        """
        embeddings = self.qi
        n_emb = np.ceil(self.num_items * limit)

        for idx, (k, v) in enumerate(item_mappings.items()):
            if idx > n_emb:
                break

            if v < self.num_items:
                embeddings[v] = pretrained_embeddings[k]

        self.qi = embeddings
        self.init_strategy = self.method
        self.init_params = self.method_args
        self.method = f'Pretrained from {ds_name}_{ds_method}'
        self.n_emb = min(n_emb, len(embeddings))
        self.limit = limit

    def _base_config(self, is_complete):
        config = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_ratings': self.num_ratings,
            'sparsity': self.dataset_sparsity,
            'bound': self.bound if hasattr(self, 'bound') else None,
            'embedding_size': self.n_factors,
            'global_bias': float(self.global_bias) if hasattr(self, 'global_bias') else None,
            'reg_factor': self.reg,
            'dataset': self.dataset_name,
            'dataset_split': self.dataset_split,
            'method': self.method,
            'method_args': self.method_args,
            'epochs': self.epochs,
            'last_epoch': self.last_epoch,
            'learning_rate': self.lr,
            'patience': self.patience,
            'results': self.results,
            'n_emb': self.n_emb,
            'limit': self.limit,
            'init': self.init_strategy,
            'init_params': self.init_params,
            'is_complete': is_complete
        }

        return config

    def _save_config(self, path, is_complete):
        """
        Saves model attributes to a config file.
        """

        try:
            config = self._base_config(is_complete)
        except Exception as e:
            raise Exception('Please call .fit() before trying to save.', e)

        io.save_json(path, 'config', config, **{'indent': 4})

    @abstractmethod
    def _save_embeddings(self, path):
        """
        Saves model embeddings to .npy files.
        """
        pass

    def save(self, is_complete=True):
        """
        Saves model embeddings and attributes to files. If plus is not None,
        the model will be saved inside the 'models/{plus}/' folder.
        """
        if (self.model_path is None):
            raise UndefinedOutputException()

        path = self.model_path
        self._save_config(path, is_complete)
        self._save_embeddings(path)

    @abstractmethod
    def load_embeddings(self, path):
        """
        Loads embeddings from .npy files inside the given path.
        """
        pass

    def load_attributes(self, config):
        """
        Loads class attributes from the config dictionary.
        """
        self.global_bias = config['global_bias']
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_ratings = config['num_ratings']
        self.dataset_name = config['dataset']
        self.dataset_split = config['dataset_split']
        self.dataset_sparsity = config['sparsity']
        self.lr = config['learning_rate']
        self.epochs = config['epochs']
        self.patience = config['patience']
        self.last_epoch = config['last_epoch']
        self.results = config['results']
        self.init_strategy = config['init']
        self.init_params = config['init_params']
        self.limit = config['limit']
        self.n_emb = config['n_emb']

    @staticmethod
    def _load_config(path):
        config = io.load_json(path, 'config')
        return config

    @abstractstaticmethod
    def load_model(path):
        """
        Loads model from a given path.
        """
        pass

    def _create_model_folder(self, model_path):
        path = model_path
        now = datetime.now().isoformat()
        folder_name = f'{self.dataset_name}_{now}'
        io.create_folder(path, folder_name)

        path += f'/{folder_name}'
        self.model_path = path

        io.save_json(self.model_path, 'config', {
            'is_complete': False
        })
