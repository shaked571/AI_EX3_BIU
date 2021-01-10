import pandas as pd
import os
import numpy as np


class DataHandler:

    def __init__(self, base_data_path):
        self.min_count = 7  # calculated by taking the top 0.9 percentage of te counting
        self.books_data: pd.DataFrame = pd.read_csv(os.path.join(base_data_path, "books.csv"), encoding='ISO-8859-1')
        self.books_data_index_book_id = self.books_data.set_index(keys=['book_id'], drop=True)
        self.users_data = pd.read_csv(os.path.join(base_data_path, "users.csv"))
        self.ratings_data = pd.read_csv(os.path.join(base_data_path, "ratings.csv"), encoding='ISO-8859-1')
        self.user_rating = self.merge_users_and_ratings(self.users_data, self.ratings_data)
        self.general_ratings_matrix: pd.DataFrame = self.prepare_rating_matrix(self.ratings_data)
        self.C_total_mean = self.general_ratings_matrix['avg'].mean()

    def id2title(self, book_id) -> str:
        return self.books_data_index_book_id.loc[book_id]['title']

    def prepare_rating_matrix(self, ratings_data: pd.DataFrame) -> pd.DataFrame:
        ratings_data = ratings_data.reset_index().filter(items=['book_id', 'user_id', 'rating'])
        pivot_rd = ratings_data.pivot_table(index=['book_id'], columns=['user_id'])
        v = pivot_rd.count(axis=1)
        r = pivot_rd.mean(axis=1)
        pivot_rd['avg'] = r
        pivot_rd['vote_count'] = v
        return pivot_rd[pivot_rd.vote_count >= self.min_count]

    def get_rating_table_by_age(self, df, age):
        low_bound = (age // 10) * 10 + 1
        high_bound = ((age // 10) + 1) * 10
        return df[(df['age'] >= low_bound) & (df['age'] <= high_bound)]

    def get_rating_table_by_location(self, df, loc):
        return df[df['location'] == loc]

    def merge_users_and_ratings(self, users: pd.DataFrame, ratings: pd.DataFrame):
        users = users.set_index('user_id')
        ratings = ratings.set_index('user_id')
        ratings['age'] = np.nan
        ratings['location'] = np.nan
        ratings.update(users)
        return ratings


if __name__ == '__main__':
    dh = DataHandler("data")
    a = 2
    print()
