import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz.distance import Hamming, Indel, Jaro, JaroWinkler, Levenshtein

from .cosine import cosine_similarity

mapping = {
    'cosine': cosine_similarity,
    # 'hamming': Hamming.similarity,
    'indel': Indel.similarity,
    'jaro': Jaro.similarity,
    'jaro_winkler': JaroWinkler.similarity,
    'levenshtein': Levenshtein.similarity
}


def matching_by_name(name):
    if (name in mapping):
        return mapping[name]
    
    raise NotImplementedError('Similarity metric is not yet implemented.')


def evaluate_distances(sentence, options, method):
    sim = matching_by_name(method)

    distances = np.array([
        sim(sentence, opt) for opt in options
    ])

    if (method in ['indel', 'levenshtein']):
        scaler = MinMaxScaler()
        distances = scaler.fit_transform(distances.reshape(-1, 1))
        distances = distances.flatten()

    return distances


def match_to_options(sentence, options, method='indel'):
    distances = evaluate_distances(sentence, options, method)
    
    best_index = np.argmax(distances)
    best_value = distances[best_index]

    return best_index, best_value


def match(sentence, options, weights=None):
    if (weights is None):
        weights = np.ones(len(mapping))

    values = np.zeros(len(options))

    for i, m in enumerate(mapping.keys()):
        distances = evaluate_distances(sentence, options, m)
        distances = distances * weights[i]
        values += distances

    values /= np.sum(weights)
    best_index = np.argmax(values)
    best_value = values[best_index]

    return best_index, best_value
