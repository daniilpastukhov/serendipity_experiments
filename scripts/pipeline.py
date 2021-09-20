# preprocess
# find model
# evaluate model

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.metaclient import MetadataClient
from utils.config_parser import parse_config
from utils.dataset_props import DatasetProps, get_prop_names, get_dict_names
from scripts.models3 import train_ensemble

logger = logging.getLogger(__name__)
# config = parse_config('config/movielens100k.ini')
config = parse_config('config/movielens1m.ini')


def compare_scores(client, props, algorithm):
    metadata = list(client.get_scores())  # get meta data from meta storage
    df = pd.DataFrame(metadata, columns=get_dict_names())
    df = df[df['model'] == algorithm]  # filter out irrelevant models
    props_df = df[get_prop_names()]
    print([props.get_props(to_list=True)], props_df.values)
    similarities = cosine_similarity([props.get_props(to_list=True)], props_df.values)[0]
    max_ind = np.argmax(similarities)
    if similarities[max_ind] < 0.8:
        return None
    else:
        return df.iloc[max_ind]['best_params']


metadata_client = MetadataClient()
ratings = pd.read_csv(config['paths']['RATINGS_PATH'])
items = pd.read_csv(config['paths']['ITEMS_PATH'])
dataset_name = 'movielens-1m'
dataset_props = DatasetProps(dataset_name=dataset_name, ratings=ratings, items=items)

algorithm_name = '3-models'
# metadata_client.add_dataset(dataset_props.get_dict(algorithm_name, [1, 95, 48]))

params = compare_scores(metadata_client, dataset_props, algorithm_name)
res = None

if params is not None:
    logger.debug("Using meta-knowledge to initialize the population.")
    res = train_ensemble(params)
else:
    logger.debug("Using random population sampling.")
    res = train_ensemble()

