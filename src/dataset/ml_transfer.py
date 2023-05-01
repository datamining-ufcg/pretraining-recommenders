from surprise import Reader

from src.dataset.movielens import ExplicitMovielens, ImplicitMovielens, Movielens


class MLTransfer(Movielens):
    def __init__(
        self,
        dataset_foldername,
        ratings_filename,
        items_filename,
        items_separator,
        is_source,
        **kwargs
    ) -> None:
        self.dataset_foldername = dataset_foldername
        self.ratings_filename = ratings_filename
        self.items_filename = items_filename
        self.items_separator = items_separator
        self.is_source = is_source
        if (is_source):
            self.current_fold = 1
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
        if (self.ratings_filename.endswith('.csv')):
            return self.ratings_filename
        return f'{self.ratings_filename}.csv'

    @property
    def items_file(self):
        return self.items_filename

    def load_item_description(self):
        self.check_and_preprocess_items()
        path = f'{self.ds_path}/{self.items_file}'
        ends = ['20m', '20m_negative', '25m', '25m_negative']
        adjust = not any([
            self.dataset_foldername.endswith(e) for e in ends
        ])
        return super().load_item_description(
            path,
            {'sep': self.items_separator},
            adjust_cols=adjust
        )


class ExplicitMLTransfer(MLTransfer, ExplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ExplicitMLTransfer'
        kwargs['name'] = self.name
        super(
            ExplicitMLTransfer,
            self
        ).__init__(**kwargs)


class ImplicitMLTransfer(MLTransfer, ImplicitMovielens):
    """

    """

    def __init__(self, **kwargs) -> None:
        self.name = 'ImplicitMLTransfer'
        kwargs['name'] = self.name
        super(
            ImplicitMLTransfer,
            self
        ).__init__(**kwargs)
