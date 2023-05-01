from surprise import Reader
from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens


class ML1M():
    """
    Implementation of ML1M with Surprise.
    """

    @property
    def ds_path(self):
        return f'{self.data_path}/ml1m'

    @property
    def sep(self):
        return '::'

    @property
    def reader(self):
        return Reader('ml-1m', rating_scale=(1, 5))

    @property
    def ratings_file(self):
        return 'ratings.dat'

    @property
    def items_file(self):
        return 'movies.dat'

    def load_item_description(self):
        self.check_and_preprocess_items()
        read_params = {'sep': self.sep}
        path = f'{self.ds_path}/{self.items_file}'
        return super().load_item_description(path, read_params)


class ExplicitML1M(ML1M, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitML1M'
        super(ExplicitML1M, self).__init__(self.name, **kwargs)


class ImplicitML1M(ML1M, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitML1M'
        super(ImplicitML1M, self).__init__(self.name, **kwargs)
