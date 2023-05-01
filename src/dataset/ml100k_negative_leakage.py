from surprise import Reader

from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens, Movielens


class ML100kNegativeLeakage(Movielens):
    def __init__(self, dataset_foldername, items_filename, items_separator, **kwargs) -> None:
        self.dataset_foldername = dataset_foldername
        self.items_filename = items_filename
        self.items_separator = items_separator
        super().__init__(**kwargs)

    @property
    def ds_path(self):
        return f'{self.data_path}/{self.dataset_foldername}'

    @property
    def sep(self):
        return ','

    @property
    def reader(self):
        rdr = Reader(
            line_format='user item rating timestamp',
            sep=self.sep, rating_scale=(1, 5),
            skip_lines=1
        )
        return rdr

    @property
    def ratings_file(self):
        return 'negative_ml100k.csv'

    @property
    def items_file(self):
        return self.items_filename

    def load_item_description(self):
        self.check_and_preprocess_items()
        path = f'{self.ds_path}/{self.items_file}'
        adjust = not (self.items_filename.endswith('20m')
                      or self.items_filename.endswith('25m'))
        return super().load_item_description(
            path, {'sep': self.items_separator}, adjust_cols=adjust)


class ExplicitML100kNegativeLeakage(ML100kNegativeLeakage, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitML100kNegativeLeakage'
        kwargs['name'] = self.name
        super(
            ExplicitML100kNegativeLeakage,
            self
        ).__init__(**kwargs)


class ImplicitML100kNegativeLeakage(ML100kNegativeLeakage, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitML100kNegativeLeakage'
        kwargs['name'] = self.name
        super(
            ImplicitML100kNegativeLeakage,
            self
        ).__init__(**kwargs)
