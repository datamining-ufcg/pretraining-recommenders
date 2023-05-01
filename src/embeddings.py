import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD as PCA
from gensim.models import Word2Vec

from src.config import RANDOM_STATE


class EmbeddingGenerator:
    """
    Auxiliar class for embedding generation.
    """

    @staticmethod
    def get_embeddings(
        trainset,
        embedding_dim,
        method='random',
        user=False,
        **kwargs
    ):
        if (method == 'word2vec' and user):
            print('User embeddings cannot be generated with Word2Vec.')
            print('Reverting to random initialization.')
            method = 'random'
            kwargs = {'stddev': 0.1}

        if (method == 'random'):
            size = trainset.n_items if not user else trainset.n_users
            embeddings = EmbeddingGenerator.get_random_embeddings(
                size, embedding_dim, **kwargs)

        elif (method == 'pca'):
            embeddings = EmbeddingGenerator.get_pca_embeddings(
                trainset, embedding_dim, user)

        elif (method == 'word2vec'):
            embeddings = EmbeddingGenerator.get_word_embeddings(
                trainset, embedding_dim, **kwargs)

        if (isinstance(embeddings[0][0], np.float64)):
            embeddings = embeddings.astype(np.float32)

        return embeddings

    @staticmethod
    def get_random_embeddings(size, embedding_dim, stddev, use_sigma=False):
        """
        Generate a matrix of embeddings following a normal distribution N(0, $\sigma$), where $\sigma$ is stddev.
        If `use_sigma` is True, then it will calculate the standard deviation $\sigma$ by $\frac{1}{\sqrt{size}} \cdot \sigma^{*}$
        """
        sigma = stddev * (1 / np.sqrt(size))
        sigma = sigma if use_sigma else stddev
        embeddings = np.random.normal(
            0, sigma, (size, embedding_dim)
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def get_word_embeddings(trainset, embedding_dim, window_size, use_sg=True, **kwargs):
        """
        Retrieves item embeddings using Word2Vec strategies from Gensim.
        """

        sentences = []
        for uid in trainset.all_users():
            sentences.append([str(iid) for iid, _ in trainset.ur[uid]])

        sg = 1 if use_sg else 0
        wv = Word2Vec(sentences, vector_size=embedding_dim,
                      window=window_size, sg=sg, min_count=1)
        qi = np.array([wv.wv[str(iid)] for iid in trainset.all_items()])

        return qi

    @staticmethod
    def get_item2vec(trainset, embedding_dim, use_sg, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def build_ratings_matrix(trainset, testset=None):
        """
        Builds ratings matrix using numpy array.
        """
        ratings_matrix = np.zeros(
            (trainset.n_users, trainset.n_items),
            dtype=np.float32
        )

        _data_source = testset if testset is not None else trainset.all_ratings()
        for (u, i, r) in _data_source:
            ratings_matrix[u][i] = r

        return ratings_matrix

    @staticmethod
    def build_csr_ratings_matrix(trainset):
        """
        Builds ratings matrix using sparse matrix.
        """
        user = []
        item = []
        ratings = []

        for (u, i, r) in trainset.all_ratings():
            user.append(u)
            item.append(i)
            ratings.append(r)

        ratings_matrix = csr_matrix(
            (ratings, (user, item)),
            shape=(trainset.n_users, trainset.n_items)
        )

        return ratings_matrix

    @staticmethod
    def _build_user_item_matrix(trainset):
        if (trainset.n_ratings <= 15 * 1e6):
            ratings_matrix = EmbeddingGenerator.build_ratings_matrix(trainset)
        else:
            ratings_matrix = EmbeddingGenerator.build_csr_ratings_matrix(
                trainset)

        return ratings_matrix

    @staticmethod
    def get_pca_embeddings(trainset, embedding_dim, user):
        input_matrix = EmbeddingGenerator._build_user_item_matrix(trainset)
        if not user:
            input_matrix = input_matrix.T

        norm_matrix = normalize(input_matrix, axis=1)
        reducer = PCA(n_components=embedding_dim, random_state=RANDOM_STATE)
        emb = normalize(reducer.fit_transform(norm_matrix), axis=1)
        return emb.astype(np.float32)
