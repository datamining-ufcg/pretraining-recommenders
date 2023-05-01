from src.dataset.dataset import Dataset


class ExplicitFeedback(Dataset):
    def __init__(self, name, **kwargs) -> None:
        super(ExplicitFeedback, self).__init__(name, **kwargs)

    @property
    def is_loaded(self):
        return hasattr(self, 'data')

    def check_preprocessed(self):
        return self.check_file(self.ratings_file)

    def check_ratings_intermediate_file(self):
        return self.check_file(self.ratings_intermediate_file)

    def get_embeddings(self, embedding_dim, method='random', user=False, **kwargs):
        return super().get_embeddings(False, embedding_dim, method, user, **kwargs)
