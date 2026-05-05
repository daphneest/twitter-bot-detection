from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Any

from config import DATA_DIR


FEATURES = [
    'statuses_count', 'followers_count', 'friends_count', 'favourites_count',
    'listed_count', 'default_profile', 'default_profile_image', 'geo_enabled',
    'profile_use_background_image', 'verified',
    'name_length', 'screen_name_length', 'description_length',
    'name_contains_bot', 'screen_name_contains_bot',
    'name_entropy', 'screen_name_entropy',
    'friends_followers_ratio', 'lists_followers_ratio',
    'retweet_followers_ratio', 'favorites_followers_ratio',
    'retweet_status_ratio', 'favorites_status_ratio', 'reply_status_ratio',
    'account_age',
    'followers_account_age_ratio', 'friends_account_age_ratio',
    'statuses_account_age_ratio', 'favourites_account_age_ratio',
    'lists_account_age_ratio',
]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / 'twitter_bots.csv')
    df = df.fillna(0)
    X = df[FEATURES]
    y = df['class_bot']
    return tuple(train_test_split(X, y, test_size=0.2, random_state=42, stratify=y))
