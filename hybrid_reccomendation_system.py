from pyspark import SparkContext
import sys
import os
import math
import time
import heapq
from itertools import combinations
import pickle
from prep_data_for_model import prep_data_for_model

# Configure PySpark to use a specific Python version based on the command line argument
os.environ['PYSPARK_PYTHON'] = sys.argv[2]

# Initialize Spark context for distributed computing
sc = SparkContext('local[*]', 'recommend')


def collaborative_filtering_recommendation():
    """
    Generates item recommendations using collaborative filtering on user-item ratings data.
    """

    def return_paired_ratings(input_rdd):
        """
        Generates all pairwise combinations of item ratings for each user.

        Parameters:
        - input_rdd: RDD of user ratings (user, [(item1, rating1), (item2, rating2), ...])

        Returns:
        - List of tuples with format (((item1, item2), (rating1, rating2)), ...)
        """
        return_list = []

        for user, ratings in input_rdd:
            ratings = list(ratings)

            # Generate all pairs of ratings for each user
            for (item1, rate1), (item2, rate2) in combinations(ratings, 2):
                # Order the pairs to maintain consistency
                if item1 < item2:
                    ret_tuple = ((item1, item2), (rate1, rate2))
                else:
                    ret_tuple = ((item2, item1), (rate2, rate1))

                return_list.append(ret_tuple)

        return return_list

    def calculate_pearson_correlation(rdd):
        """
        Calculates the Pearson correlation coefficient for item pairs.

        Parameters:
        - rdd: RDD of item pairs and their ratings

        Returns:
        - Tuple of item pair and their Pearson correlation
        """
        i_sum = j_sum = 0
        ratings = list(rdd[1])

        for i, j in ratings:
            i_sum += i
            j_sum += j

        avg_i = i_sum / len(ratings)
        avg_j = j_sum / len(ratings)

        numerator_sum = denominator_i_sum = denominator_j_sum = 0

        for i, j in ratings:
            numerator_sum += ((i - avg_i) * (j - avg_j))
            denominator_i_sum += ((i - avg_i) ** 2)
            denominator_j_sum += ((j - avg_j) ** 2)

        if numerator_sum != 0:
            correlation = numerator_sum / (math.sqrt(denominator_i_sum) * math.sqrt(denominator_j_sum))
            return (rdd[0], correlation)
        else:
            return (rdd[0], 0)

    def calc_estimation(input_rdd, users_to_business_ratings_dict, pearson_correlations, avg_item_rating):
        """
        Estimates ratings for items in the test dataset using collaborative filtering.

        Parameters:
        - input_rdd: RDD of user, item, and actual rating
        - users_to_business_ratings_dict: Dictionary of user ratings
        - pearson_correlations: Broadcasted Pearson correlations
        - avg_item_rating: Average rating per item

        Returns:
        - List of tuples with estimated ratings
        """
        return_list = []
        top_k = 10  # Number of top similar items to consider

        for user, business, rating in input_rdd:
            numerator_sum = denominator_sum = user_sum = 0
            min_heap = []

            for b, r in users_to_business_ratings_dict[user]:
                pair = (min(business, b), max(business, b))
                if pair in pearson_correlations:
                    if len(min_heap) < top_k:
                        heapq.heappush(min_heap, (pearson_correlations[pair], r))
                    elif pearson_correlations[pair] > min_heap[0][0]:
                        heapq.heappop(min_heap)
                        heapq.heappush(min_heap, (pearson_correlations[pair], r))

                user_sum += r

            for item in min_heap:
                numerator_sum += (item[0] * item[1])
                denominator_sum += abs(item[0])

            if numerator_sum == 0 and business in avg_item_rating:
                avg = avg_item_rating[business] + user_sum / len(users_to_business_ratings_dict[user])
                return_list.append((user, business, rating, avg / 2))
            elif numerator_sum == 0:
                return_list.append((user, business, rating, user_sum / len(users_to_business_ratings_dict[user])))
            else:
                estimation = numerator_sum / denominator_sum
                return_list.append((user, business, rating, estimation))

        return return_list

    # Set number of partitions for parallel processing
    n_partitions = 50
    # Load and preprocess training data
    ratings_rdd = sc.textFile(sys.argv[1] + '/yelp_train.csv', n_partitions)
    header = ratings_rdd.first()  # Remove header
    ratings_rdd = ratings_rdd.filter(lambda line: line != header).map(lambda line: line.split(',')).map(
        lambda line: [line[0], line[1], int(line[2][0])]).cache()

    # Calculate average rating per business
    average_rating_per_item = ratings_rdd.map(lambda x: (x[1], x[2])).mapValues(lambda v: (v, 1)).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(lambda v: v[0] / v[1]).collectAsMap()

    # Load and preprocess test data
    test_rdd = sc.textFile(sys.argv[1] + 'yelp_val.csv', n_partitions)
    header = test_rdd.first()  # Remove header
    test_rdd = test_rdd.filter(lambda line: line != header).map(lambda line: line.split(',')).map(
        lambda line: [line[0], line[1], int(line[2][0])]).cache()

    # Prepare user-item ratings for collaborative filtering
    users_to_business_ratings = ratings_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().cache()
    pairs_of_ratings = users_to_business_ratings.mapPartitions(lambda x: return_paired_ratings(x))

    # Calculate Pearson correlations and filter by threshold
    min_co_ratings = 40  # Minimum co-ratings to consider for correlation
    pearson_correlations_rdd = pairs_of_ratings.groupByKey().filter(lambda x: len(x[1]) >= min_co_ratings).map(
        calculate_pearson_correlation).filter(lambda x: x[1] > 0.1)
    pearson_correlations_broadcast = sc.broadcast(pearson_correlations_rdd.collectAsMap())

    users_to_business_ratings_dict = users_to_business_ratings.map(lambda x: (x[0], list(x[1]))).collectAsMap()

    # Estimate scores for test dataset
    estimated_scores = test_rdd.mapPartitions(
        lambda x: calc_estimation(x, users_to_business_ratings_dict, pearson_correlations_broadcast.value,
                                  average_rating_per_item))

    return estimated_scores.collect()


def model_based_recommendation():
    """
    Generates item recommendations using a pre-trained machine learning model.

    Returns:
    - Tuple of model predictions and test dataframe
    """
    # Prepare data for model
    train_df, test_df, attributes = prep_data_for_model(sc, data_path='data')
    # Load pre-trained model
    with open('models/final_xgb_model.pickle', 'rb') as file:
        model = pickle.load(file)
    model_test_pred = model.predict(test_df[attributes])

    return model_test_pred, test_df


start = time.time()

# Define weights for combining recommendations
collaborative_filtering_weight = 0.2
model_based_weight = 1 - collaborative_filtering_weight

# Generate recommendations
collaborative_filtering_result = collaborative_filtering_recommendation()
model_based_result, test_df = model_based_recommendation()

# Calculate RMSE for combined recommendations
big_sum = count = 0
for i in range(len(model_based_result)):
    estimate = (model_based_result[i] * model_based_weight) + (
                collaborative_filtering_result[i][3] * collaborative_filtering_weight)
    actual_rating = test_df['true_rating'][i]
    big_sum += ((round(estimate, 2) - actual_rating) ** 2)
    count += 1

# Output performance metrics
print(f'RMSE: {math.sqrt((1 / count) * big_sum)}')
print(f'DURATION: {time.time() - start}')
