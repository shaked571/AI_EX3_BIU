import heapq
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from math import sqrt
from data_handler import DataHandler
from functools import lru_cache


class RecommendationSystem:

    def __init__(self, base_data_path: str = 'data'):
        self.dh = DataHandler(base_data_path)
        # self.CF = None
        # self.users_mean = None
        # self.ratings_diff = None
        self.data_matrix = None

    def weighted_average(self, row):
        return ((row['vote_count'] / (row['vote_count'] + self.dh.min_count)) * row['avg']) + (
                (self.dh.min_count / (self.dh.min_count + row['vote_count'])) * self.dh.C_total_mean)

    def get_simply_recommendation(self, k: int) -> List[Tuple[str, int, float]]:
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_ratings_matrix = self.dh.general_ratings_matrix
        return self.enrich_rating_tables(k, user_ratings_matrix)

    def get_simply_age_recommendation(self, age, k):
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_rating = self.dh.user_rating
        user_rating = self.dh.get_rating_table_by_age(user_rating, age)
        user_ratings_matrix = self.dh.prepare_rating_matrix(user_rating)
        return self.enrich_rating_tables(k, user_ratings_matrix)

    def get_simply_place_recommendation(self,loc: str, k: int) -> List[Tuple[str, int, float]]:
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_rating = self.dh.user_rating
        user_rating = self.dh.get_rating_table_by_location(user_rating, loc)
        user_ratings_matrix = self.dh.prepare_rating_matrix(user_rating)
        return self.enrich_rating_tables(k, user_ratings_matrix)

    def enrich_rating_tables(self, k, user_ratings_matrix) -> List[Tuple[str, int, float]]:
        user_ratings_matrix['w_avg'] = user_ratings_matrix.apply(self.weighted_average, axis=1)
        top_general_pick = user_ratings_matrix.sort_values(by='w_avg', ascending=False)['w_avg']
        top_k_ids = list(top_general_pick[:k].index.get_level_values(0))
        top_k_title = [self.dh.id2title(idx) for idx in top_k_ids]
        top_k_scores = list(top_general_pick[:k])
        return list(zip(top_k_title, top_k_ids, top_k_scores))

    def build_CF_prediction_matrix(self, sim):
        if sim not in ['jaccard', 'cosine', 'euclidean']:
            raise ValueError("We support only the following types: 'jaccard', 'cosine', 'euclidean'")
        ratings_diff, users_mean, self.data_matrix = self.dh.prepare_norm_user_rating_matrix(self.dh.ratings_data)
        user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
        user_similarity = np.array([self.keep_top_k(np.array(arr), 20) for arr in user_similarity])
        self.pred = users_mean + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        return user_similarity

    def keep_top_k(self, arr, k):
        smallest = heapq.nlargest(k, arr)[-1]
        arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
        return arr

    def get_CF_recommendation(self, user_id, k):
        if self.pred is None:
            raise ValueError("Need first to build the CF matrix using 'build_CF_prediction_matrix()' ")
        user_id = user_id - 1
        predicted_ratings_row = self.pred[user_id]
        data_matrix_row = self.data_matrix[user_id]

        # print("Top rated books by test user:")
        # print(self.get_top_rated(data_matrix_row, k))

        # print('****** test user - user_prediction ******')
        recommendations = self.get_recommendations(predicted_ratings_row, data_matrix_row, k)
        # print(recommendations)
        return recommendations

    def get_top_rated(self, data_matrix_row, k):
        srt_idx = np.argsort(-data_matrix_row)
        srt_idx_not_nan = srt_idx[~np.isnan(data_matrix_row[srt_idx])]
        top_k_title = [self.dh.id2title(self.dh.id2xbookid[idx]) for idx in srt_idx_not_nan][:k]
        return top_k_title

    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(self, predicted_ratings_row, data_matrix_row, k):

        predicted_ratings_unrated = predicted_ratings_row[np.isnan(data_matrix_row)]

        book_ids = np.argsort(-predicted_ratings_unrated)[:k]
        books_rating = np.sort(predicted_ratings_unrated)[::-1][:k]

        # Return top k movies
        return [(self.dh.id2title(self.dh.id2xbookid[idx]), self.dh.id2xbookid[idx], rating) for idx, rating in zip(book_ids, books_rating)]

    def get_sorted_recommendations_from_cf(self, user_id):
        user_id = user_id - 1
        predicted_ratings_row = self.pred[user_id]
        data_matrix_row = self.data_matrix[user_id]
        predicted_ratings_unrated = predicted_ratings_row[np.isnan(data_matrix_row)]

        book_ids = np.argsort(-predicted_ratings_unrated)
        books_rating = np.sort(predicted_ratings_unrated)[::-1]

        return {idx: rating for idx, rating in zip(book_ids, books_rating)}

    def build_contact_sim_matrix(self):
        suffix_books_feature = self.build_tags_features() # numpy size(num_of_books, len(common_tags))
        books_data = self.dh.books_data
        prefix_books_feature = self.build_other_features() # numpy size(num_of_books, ???)
        books_matrix = self.merge_faetures(suffix_books_feature, prefix_books_feature)
        books_matrix = pairwise_distances # use the last section pairwise builder

    def find_common_tags(self, books_tags) -> List[int]:
        '''
        Get a Dataframe of books and tags and return the tags that appear at list twice
        :param books_data:
        :return:
        '''
        count_tag = Counter(books_tags['tag_id'].to_list())
        common_tags = {x: count for x, count in count_tag.items() if count > 1}
        return list(common_tags.keys())

    def build_tags_features(self):
        books_tags = self.dh.books_tags
        common_tags = self.find_common_tags(books_tags)
        tag2feature = {tag: i for i, tag in enumerate(common_tags)}
        vectors = []
        for book_id, tags in books_tags.groupby(by='goodreads_book_id'):
            vec = np.zeros(len(common_tags))
            for t in tags.iterrows():
                t_id = t[1].tag_id
                if t_id in tag2feature:
                    vec[tag2feature[t_id]] = 1
            vectors.append(vec)
        return np.array(vectors)

    def build_other_features(self):
        pass

    def merge_faetures(self, suffix_books_feature, prefix_books_feature):
        pass

    @staticmethod
    def high_rating(rating):
        if rating > 3:
            return True
        return False

    def filter_test(self, k):
        relevant_users = {}
        test = self.dh.test
        test["is_high"] = test["rating"].apply(self.high_rating)
        for user_id, group_df in test.groupby(by="user_id"):
            if len(group_df[group_df['is_high']]) >= k:
                relevant_books = list(group_df[group_df['is_high']]["book_id"])
                relevant_users[user_id] = relevant_books
        return relevant_users

    def precision_k(self, k):
        print("precision_k")
        relevant_users = self.filter_test(k)
        for sim in ["cosine", "euclidean", "jaccard"]:
            self.build_CF_prediction_matrix(sim)
            hits = 0
            for user_id, high_rated_books in relevant_users.items():
                recommendations = self.get_CF_recommendation(user_id, k)
                for (_, book_id, _) in recommendations:
                    if book_id in high_rated_books:
                        hits += 1
            precision = round(hits/(k*len(relevant_users)), 3)
            print(f"Accuracy with similarity {sim} is {precision}")

    def ARHR(self, k):
        print("ARHR")
        relevant_users = self.filter_test(k)
        for sim in ["cosine", "euclidean", "jaccard"]:
            self.build_CF_prediction_matrix(sim)
            hits = 0
            for user_id, high_rated_books in relevant_users.items():
                recommendations = self.get_CF_recommendation(user_id, k)
                for i, (_, book_id, _) in enumerate(recommendations):
                    if book_id in high_rated_books:
                        hits += 1/(i+1)
            arhr = round(hits / len(relevant_users), 3)
            print(f"Accuracy with similarity {sim} is {arhr}")

    def RMSE(self):
        print("RMSE")
        for sim in ["cosine", "euclidean", "jaccard"]:
            self.build_CF_prediction_matrix(sim)
            sum_error = 0
            count_lines = 0
            for user_id, group_df in self.dh.test.groupby(by="user_id"):
                predicted_recs = self.get_sorted_recommendations_from_cf(user_id)
                for row in group_df.itertuples(index=False):
                    _, book_id, rating = tuple(row)
                    predicted_rating = predicted_recs[book_id] if book_id in predicted_recs else 0
                    sum_error += (predicted_rating - rating)**2
                    count_lines += 1
            rmse = round(sqrt(sum_error/count_lines), 3)
            print(f"Accuracy with similarity {sim} is {rmse}")


if __name__ == '__main__':
    rc = RecommendationSystem()
    # print(rc.get_simply_recommendation(10))
    # print(rc.get_simply_place_recommendation('Ohio', 10))
    # print(rc.get_simply_age_recommendation(28, 10))
    # rc.build_CF_prediction_matrix('cosine')
    # rec = rc.get_CF_recommendation(511, 10)
    # print(rec)
    # rc.build_contact_sim_matrix()
    rc.RMSE()
    # rc.ARHR(10)
    # jac = rc.build_CF_prediction_matrix('jaccard')
    # euc = rc.build_CF_prediction_matrix( 'euclidean')
    # res = rc.user_ratings_matrix['w_avg'].iloc[500]
    #
    # r = rc.user_ratings_matrix['avg'].iloc[500]
    # v = rc.user_ratings_matrix['vote_count'].iloc[500]
    # C = rc.dh.C_total_mean
    # m = rc.dh.min_count

