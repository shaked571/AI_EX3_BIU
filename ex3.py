import heapq
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from math import sqrt
from data_handler import DataHandler
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.options.display.max_colwidth = 100


class RecommendationSystem:

    def __init__(self, base_data_path: str = 'data'):
        self.dh = DataHandler(base_data_path)
        self.data_matrix = None

    def weighted_average(self, row):
        """
        calculate the weighted average of a user
        :param row: a user representation
        :return:
        """
        return ((row['vote_count'] / (row['vote_count'] + self.dh.min_count)) * row['avg']) + (
                (self.dh.min_count / (self.dh.min_count + row['vote_count'])) * self.dh.C_total_mean)

    def get_simply_recommendation(self, k: int) -> pd.DataFrame:
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_ratings_matrix = self.dh.general_ratings_matrix
        return self.get_top_k_from_table(k, user_ratings_matrix)

    def get_simply_age_recommendation(self, age, k):
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_rating = self.dh.user_rating
        user_rating = self.dh.get_rating_table_by_age(user_rating, age)
        user_ratings_matrix = self.dh.prepare_rating_matrix(user_rating)
        return self.get_top_k_from_table(k, user_ratings_matrix)

    def get_simply_place_recommendation(self, loc: str, k: int) -> pd.DataFrame:
        if k < 1:
            raise ValueError("k need to be a positive integer larger than 0")
        user_rating = self.dh.user_rating
        user_rating = self.dh.get_rating_table_by_location(user_rating, loc)
        user_ratings_matrix = self.dh.prepare_rating_matrix(user_rating)
        return self.get_top_k_from_table(k, user_ratings_matrix)

    def get_top_k_from_table(self, k, user_ratings_matrix) -> pd.DataFrame:
        """
        get the top k books recommendation for a yser rating matrix
        :param k: top books
        :param user_ratings_matrix: the user rating matrix
        :return: a pandas with the title id and score
        """
        user_ratings_matrix['w_avg'] = user_ratings_matrix.apply(self.weighted_average, axis=1)
        top_general_pick = user_ratings_matrix.sort_values(by='w_avg', ascending=False)['w_avg']
        top_k_ids = list(top_general_pick[:k].index.get_level_values(0)) #the pandas as multi index and the first oen iis the id
        top_k_title = [self.dh.id2title(idx) for idx in top_k_ids]
        top_k_scores = list(top_general_pick[:k])
        res = list(zip(top_k_title, top_k_ids, top_k_scores))
        plot = []
        for r in res:
            plot.append({"title": r[0], "id": r[1], "score": r[2]})

        return pd.DataFrame(plot)

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

    def get_CF_recommendation(self, user_id, k, DEBUG=False):
        if self.pred is None:
            raise ValueError("Need first to build the CF matrix using 'build_CF_prediction_matrix()' ")
        user_id = user_id - 1
        predicted_ratings_row = self.pred[user_id]
        data_matrix_row = self.data_matrix[user_id]
        if DEBUG:
            print("Top rated books by test user:")
            print(self.get_top_rated(data_matrix_row, k))

        recommendations = self.get_recommendations(predicted_ratings_row, data_matrix_row, k)
        if DEBUG:
            print('****** test user - user_prediction ******')
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
        """
        Create the features for the publish year of the books. by decades from 1850 and before by millenniums.
        :param y:
        :return:
        """
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
        """
        group the english lang together to reduce sparsity
        :param lang:
        :return:
        """
        if lang in {'en', 'eng'}:
            return 'en-US'
        return lang

    def build_other_features(self):
        """
        Build the feature from the books data file. build the features for the author, publish year and lang as one hot
        vector for all with extra weight for the author
        :return:
        """
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
        """
        merge the 2 feature vectors created in to 1
        """
        vecs = []
        suffix_feature_num = len(list(suffix_books_feature.values())[0])
        for book_id, vec_prefix in prefix_books_feature.items():  # we know that the prefix have all the books ids
            if book_id in suffix_books_feature:
                vec = np.concatenate([vec_prefix, suffix_books_feature[book_id]], axis=0)
            else:
                vec = np.concatenate([vec_prefix, np.zeros(suffix_feature_num)], axis=0)
            vecs.append(vec)
        return np.array(vecs)

    @staticmethod
    def high_rating(rating):
        if rating > 3:
            return True
        return False

    def filter_test(self, k):
        """
        filter the test set only for users that rated movies as score of at list 4 at list k times
        """
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
                high_rated_books = set(high_rated_books)
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


def full_run():
    print(get_simply_recommendation(10))
    print(get_simply_place_recommendation('Ohio', 10))
    print(get_simply_age_recommendation(28, 10))
    build_CF_prediction_matrix('cosine')
    rec_511 = get_CF_recommendation(511, 10)
    build_contact_sim_matrix()
    get_contact_recommendation('Twilight (Twilight, #1)', 25)
    get_CF_recommendation(511, 10)
    build_contact_sim_matrix()
    precision_k(10)
    ARHR(10)
    RMSE()


rc = RecommendationSystem()
get_simply_recommendation = rc.get_simply_recommendation
get_simply_place_recommendation = rc.get_simply_place_recommendation
get_simply_age_recommendation = rc.get_simply_age_recommendation
build_CF_prediction_matrix = rc.build_CF_prediction_matrix
get_contact_recommendation = rc.get_contact_recommendation
get_CF_recommendation = rc.get_CF_recommendation
build_contact_sim_matrix = rc.build_contact_sim_matrix
precision_k = rc.precision_k
ARHR = rc.ARHR
RMSE = rc.RMSE
