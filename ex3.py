from typing import List, Tuple

from data_handler import DataHandler


class RecommendationSystem:

    def __init__(self, base_data_path: str = 'data'):
        self.dh = DataHandler(base_data_path)

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


if __name__ == '__main__':
    rc = RecommendationSystem()
    print(rc.get_simply_recommendation(10))
    print(rc.get_simply_place_recommendation('Ohio', 10))
    print(rc.get_simply_age_recommendation(28, 10))


    # res = rc.user_ratings_matrix['w_avg'].iloc[500]
    #
    # r = rc.user_ratings_matrix['avg'].iloc[500]
    # v = rc.user_ratings_matrix['vote_count'].iloc[500]
    # C = rc.dh.C_total_mean
    # m = rc.dh.min_count

