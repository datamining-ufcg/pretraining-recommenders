from surprise import Dataset as SurpriseDataset
from surprise import Reader
from surprise.model_selection import PredefinedKFold

from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens


class ML100k():
    """
    Implementation of ML100k with Surprise.
    """

    @property
    def ds_path(self):
        return f'{self.data_path}/ml100k'

    @property
    def sep(self):
        return '\t'

    @property
    def ratings_file(self):
        return ''

    @property
    def items_file(self):
        return 'u.item'

    @property
    def n_items(self):
        return 1682

    @property
    def reader(self):
        return Reader('ml-100k', rating_scale=(1, 5))

    def kfold(self, k=5):
        kfold = PredefinedKFold()
        return kfold.split(self.data)

    def load(self):
        train_file = f'{self.ds_path}/u%d.base'
        test_file = f'{self.ds_path}/u%d.test'

        folds_files = [(train_file % i, test_file % i) for i in range(1, 6)]
        self.data = SurpriseDataset.load_from_folds(
            folds_files, reader=self.reader)

        return self.data

    def load_item_description(self):
        self.check_and_preprocess_items()
        read_params = {'sep': '|'}
        path = f'{self.ds_path}/{self.items_file}'
        return super().load_item_description(path, read_params)


class ExplicitML100k(ML100k, ExplicitMovielens):
    """
    Implementation of ML100k with explicit feedback.
    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitML100k'
        super(ExplicitML100k, self).__init__(self.name, **kwargs)


class ImplicitML100k(ML100k, ImplicitMovielens):
    """
    Implementation of ML100k with implicit feedback.
    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitML100k'
        super(ImplicitML100k, self).__init__(self.name, **kwargs)

    def load(self):
        ML100k.load(self)
        ImplicitMovielens.load(self)

    def load_train_ratings(self):
        train_file = self.train_template.format(self.current_fold)
        filepath = f'{self.ds_path}/{train_file}'
        return super().load_train_ratings(filepath)
