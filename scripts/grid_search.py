import time
import os

import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
import logging
from datetime import date

from sklearn.model_selection import ParameterGrid

from models.primitive_models import PrimitiveModels
from models.knn_popular_optimized import KNNpopularity
from models.mf_optimized import MatrixFactorization
from models.autorec import AutoRec

from utils.metrics import serendipity
from utils.helpers import get_movies_by_ids, get_control_items, get_movies_by_profile
from utils.parser import parse_args
from utils.logging import init_logging

import warnings

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)

models = {
    'KNNpopularity': KNNpopularity,
    'MatrixFactorization': MatrixFactorization,
    'AutoRec': AutoRec
}

args = parse_args()

init_logging('gs.log', path=args.save_path)

# Datasets
DATASET = '1m'
DATA_PATH = 'data/movielens/' + DATASET + '/clean/'

ratings = pd.read_csv(DATA_PATH + 'ratings.csv')
movies = pd.read_csv(os.path.join(DATA_PATH, 'movies.csv'))

train_data = pd.read_csv(DATA_PATH + 'train_data.csv', index_col='userId')
test_df = pd.read_csv(DATA_PATH + 'test_data.csv', index_col='userId')
test_data, control_items = get_control_items(ratings, test_df)

user_embeddings = {}
for user_id, user_profile in test_data.iterrows():
    user_embeddings[user_id] = get_movies_by_profile(movies, user_profile)

logging.info('Dataset: ' + DATASET)

item_ratings = load(DATA_PATH + 'item_sum_dif_rating.pickle')

primitive_models = PrimitiveModels(train_data, ratings, item_ratings)
n = 10  # number of recommendations for each user

primitive_recommendations = None

if not os.path.isfile('cache/primitive_recommendations.joblib'):
    primitive_recommendations = primitive_models.make_recommendations(test_data, n)
    dump(primitive_recommendations, 'cache/primitive_recommendations.joblib')
else:
    primitive_recommendations = load('cache/primitive_recommendations.joblib')


def grid_search(model_name, grid, data):
    history = []

    for params in ParameterGrid(grid):
        model = models[model_name](model_name, data, item_ratings)
        model.fit(params)
        start_time = time.time()
        logging.debug('Started {}, {}'.format(model_name, params))
        recall, coverage, ser = evaluate(model, test_data)
        logging.debug('Recall: {}, coverage: {}, serendipity: {}'.format(recall, coverage, ser))
        logging.debug('Finished {}, {}'.format(model_name, params))
        time_elapsed = time.time() - start_time
        history.append((params, recall, coverage, ser, time_elapsed))

    return history


def evaluate(model, data):
    """
    Evaluate given model on the data.

    :param model: Recommendation model.
    :param data: Evaluation data.
    :return: tuple(recall, coverage, serendipity).
    """
    hit = 0  # used for recall calculation
    total_recommendations = 0
    all_recommendations = []  # used for coverage calculation

    ser = []

    for user_id, user_profile in tqdm(data.iterrows(), total=len(data)):
        prediction = model.predict(user_profile, n)
        if prediction is None or prediction.ndim == 0:
            continue

        if control_items[user_id] in prediction:  # if prediction contains control item increase hit counter
            hit += 1

        all_recommendations.extend(list(prediction))
        total_recommendations += 1

        recommended_items_embeddings = get_movies_by_ids(movies, prediction)

        if len(user_profile) == 0:  # check if user profile is empty
            continue

        ser.append(
            serendipity(recommended_items_embeddings, prediction,
                        primitive_recommendations[user_id], user_embeddings[user_id],
                        gamma=args.gamma, alpha=args.alpha, beta=args.beta)
        )

    if total_recommendations > 0:
        recall = hit / total_recommendations
    else:
        recall = 0

    coverage = np.unique(all_recommendations).shape[0] / model.train_data.shape[1]
    return recall, coverage, np.mean(ser)


params_grid = {
    'AutoRec': {'hidden_layer_size': [8, 16, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
                'random_state': [11, 42, 77]},
    'KNNpopularity': {'k': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 700, 1000]},
    'MatrixFactorization': {'n_components': [5, 10, 20, 50, 100, 200, 500, 1000, 2000], 'random_state': [42]},
}

for model_name in models.keys():
    logging.info('Grid search started: {}'.format(model_name))
    result = grid_search(model_name, params_grid[model_name], train_data)
    logging.info('Grid search ended: {}'.format(model_name))
    logging.info('Saved {} as {}'.format(model_name, os.path.join(args.save_path, model_name + '.joblib')))
    dump(result, os.path.join(args.save_path, model_name + '.joblib'))
