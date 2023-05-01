from surprise import Reader
from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens, Movielens


class ML10M(Movielens):
    """
    Implementation of ML10M with Surprise.
    """

    @property
    def ds_path(self):
        return f'{self.data_path}/ml10m'

    @property
    def sep(self):
        return '::'

    @property
    def reader(self):
        rdr = Reader(
            line_format='user item rating timestamp',
            sep=self.sep, rating_scale=(1, 5)
        )
        return rdr

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


class ExplicitML10M(ML10M, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = "ExplicitML10M"
        super(ExplicitML10M, self).__init__(self.name, **kwargs)


class ImplicitML10M(ML10M, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = "ImplicitML10M"
        super(ImplicitML10M, self).__init__(self.name, **kwargs)
