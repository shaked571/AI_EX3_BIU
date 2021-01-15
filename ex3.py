import heapq
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix

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

    def get_simply_place_recommendation(self, loc: str, k: int) -> List[Tuple[str, int, float]]:
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

        print("Top rated books by test user:")
        print(self.get_top_rated(data_matrix_row, k))

        print('****** test user - user_prediction ******')
        recommendations = self.get_recommendations(predicted_ratings_row, data_matrix_row, k)
        print(recommendations)
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

        # Return top k movies
        return [self.dh.id2title(self.dh.id2xbookid[idx]) for idx in book_ids]

    def build_contact_sim_matrix(self):
        prefix_books_feature = self.build_other_features()
        suffix_books_feature = self.build_tags_features()
        books_matrix = self.merge_features(prefix_books_feature, suffix_books_feature)
        self.pw_book_mat = 1 - pairwise_distances(books_matrix, metric='cosine')
        return self.pw_book_mat

    def get_contact_recommendation(self, book_name, k):
        mat_id = self.dh.book_id2matrix_id(self.dh.title2id(book_name))
        book_row = self.pw_book_mat[mat_id]
        best_books = np.argsort(book_row)[::-1][1:k+1]
        return [self.dh.id2title(self.dh.matrix_id2book_id(mat_id)) for mat_id in best_books]

    def find_common_tags(self, books_tags) -> List[int]:
        """
        Get a Dataframe of books and tags and return the tags that appear at list twice
        :param books_data:
        :return:
        """
        count_tag = Counter(books_tags['tag_id'].to_list())
        common_tags = {x: count for x, count in count_tag.items() if count > 2}
        return list(common_tags.keys())

    def build_tags_features(self):
        books_tags = self.dh.books_tags
        common_tags = self.find_common_tags(books_tags)
        tag2feature = {tag: i for i, tag in enumerate(common_tags)}

        vectors = {}
        for book_id, tags in books_tags.groupby(by='goodreads_book_id'):
            vec = np.zeros(len(common_tags))
            for t in tags.iterrows():
                t_id = t[1].tag_id
                if t_id in tag2feature:
                    vec[tag2feature[t_id]] = 1
            vectors[book_id] = vec
        return vectors

    @staticmethod
    def group_years(y):
        if y > 1850:
            return y // 10
        elif 1000 <= y <= 1850:
            return 100
        elif 0 < y < 1000:
            return 10
        elif y <= 0:
            return 0
        else:
            ValueError(f"Not expected input {y}")

    @staticmethod
    def group_lang(lang):
        if lang in {'en', 'eng'}:
            return 'en-US'
        return lang

    def build_other_features(self):
        books = self.dh.books_data
        #  If you would like to add an author is here
        books = books.filter(items=['book_id', 'original_publication_year', 'language_code', 'authors'])
        books['original_publication_year'] = books['original_publication_year'].fillna(value=-1).apply(self.group_years)
        books['language_code'] = books['language_code'].fillna(value='en-US').apply(self.group_lang)
        all_opt = np.concatenate([books.language_code.unique(), books.original_publication_year.unique(), books.authors.unique()], axis=0)
        opt_num = len(all_opt)
        tag2feature = {tag: i for i, tag in enumerate(all_opt)}
        extra_auther = {tag: i + len(tag2feature) for i, tag in enumerate(books.authors.unique())} #give extra value for the auther
        vectors = {}
        for i, r in books.iterrows():
            vec = np.zeros(opt_num + len(extra_auther))
            vec[tag2feature[r.language_code]] = 1
            vec[tag2feature[r.original_publication_year]] = 1
            vec[tag2feature[r.authors]] = 1
            vec[extra_auther[r.authors]] = 1

            vectors[r.book_id] = vec
        return vectors

    def merge_features(self, prefix_books_feature, suffix_books_feature):
        vecs = []
        suffix_feature_num = len(list(suffix_books_feature.values())[0])
        for book_id, vec_prefix in prefix_books_feature.items():  # we know that the prefix have all the books ids
            if book_id in suffix_books_feature:
                vec = np.concatenate([vec_prefix, suffix_books_feature[book_id]], axis=0)
            else:
                vec = np.concatenate([vec_prefix, np.zeros(suffix_feature_num)], axis=0)
            vecs.append(vec)
        return np.array(vecs)


if __name__ == '__main__':
    rc = RecommendationSystem()
    # print(rc.get_simply_recommendation(10))
    # print(rc.get_simply_place_recommendation('Ohio', 10))
    # print(rc.get_simply_age_recommendation(28, 10))
    # rc.build_CF_prediction_matrix('cosine')
    # rec_511 = rc.get_CF_recommendation(511, 10)
    rc.build_contact_sim_matrix()
    res = rc.get_contact_recommendation('Twilight (Twilight, #1)', 25)
    print(res)
    # jac = rc.build_CF_prediction_matrix('jaccard')
    # euc = rc.build_CF_prediction_matrix( 'euclidean')
    # res = rc.user_ratings_matrix['w_avg'].iloc[500]
    #
    # r = rc.user_ratings_matrix['avg'].iloc[500]
    # v = rc.user_ratings_matrix['vote_count'].iloc[500]
    # C = rc.dh.C_total_mean
    # m = rc.dh.min_count

