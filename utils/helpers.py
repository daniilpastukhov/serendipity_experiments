genres_cols = [f'feature{i + 1}' for i in range(20)]


def get_movies_by_profile(movies_df, profile):
    return movies_df[movies_df['movieId'].isin(profile[profile != 0].index.astype(int))][genres_cols].values


def get_average_genre(movies_df):
    return movies_df.mean(axis=0)[genres_cols]


def get_movies_by_ids(movies_df, ids):
    return movies_df[movies_df['movieId'].isin(ids)][genres_cols].values


def get_control_items(ratings, user_profiles=None, user_ids=None):
    """
    Get control items of test users for evaluation purposes.

    :param ratings: pd.DataFrame with columns <userId, movieId, rating, timestamp>.
    :param user_profiles: Sparse user profiles.
    :return: Tuple of pd.DataFrame with test users ratings and dictionary with control items.
    """
    if user_profiles is not None:
        user_ids = user_profiles.index

    control_items = {}

    for user_id in user_ids:
        user_ratings = ratings[ratings['userId'] == user_id]
        recent_index = user_ratings['timestamp'].idxmax()
        recent_rating = user_ratings.loc[recent_index]
        control_item = recent_rating['movieId'].astype(int)
        control_items[user_id] = control_item
        if user_profiles is not None:
            user_profiles.loc[user_id][str(control_item)] = 0.0
        else:
            ratings = ratings.drop(recent_index)

    return (ratings, control_items) if user_profiles is None else (user_profiles, control_items)


def get_user_profiles(data):
    """
    Get sparse matrix with user profiles.
    :param data: pd.DataFrame with columns <userId, movieId, rating, timestamp>.
    :return: Sparse pd.DataFrame where each row represents a user.
    """
    user_profiles = data.pivot_table(index=['userId'], columns=['movieId'], values='rating')
    user_profiles.fillna(0, inplace=True)
    return user_profiles
