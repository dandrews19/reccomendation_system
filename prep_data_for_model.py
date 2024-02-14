
import json
import pandas as pd



def prep_data_for_model(sc, data_path):
    """
    Prepares and processes data from Yelp datasets for a recommendation model.

    Parameters:
    - sc: The Spark context.
    - data_path: The file path to the dataset.

    Returns:
    - Tuple of pandas DataFrames (train_df, validation_df) and a list of attribute names.
    """

    # Set the number of partitions for RDD operations
    n_partitions = 2

    # Load and preprocess training dataset from CSV
    train_rdd = sc.textFile(data_path + '/yelp_train.csv', n_partitions)
    header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda line: line != header).map(lambda line: line.split(',')).map(
        lambda line: [line[0], line[1], int(line[2][0])]).cache()

    # Load and preprocess validation dataset from CSV
    validation_rdd = sc.textFile(data_path + '/yelp_val.csv', n_partitions)
    header = validation_rdd.first()
    validation_rdd = validation_rdd.filter(lambda line: line != header).map(lambda line: line.split(',')).map(
        lambda line: [line[0], line[1], int(line[2][0])]).cache()

    # Collect unique businesses and users from both train and validation sets
    businesses_in_train = set(train_rdd.map(lambda x: x[1]).collect())
    businesses_in_validation = set(validation_rdd.map(lambda x: x[1]).collect())
    total_businesses = businesses_in_train.union(businesses_in_validation)
    user_in_train = set(train_rdd.map(lambda x: x[0]).collect())
    user_in_validation = set(validation_rdd.map(lambda x: x[0]).collect())
    total_users = user_in_train.union(user_in_validation)

    # Load and filter business and user data based on relevance to the datasets
    business_rdd = sc.textFile(data_path + '/business.json', 50).map(
        lambda x: json.loads(x)).filter(lambda x: x['business_id'] in total_businesses).cache()
    user_rdd = sc.textFile(data_path + '/user.json', 50).map(
        lambda x: json.loads(x)).filter(lambda x: x['user_id'] in total_users).cache()

    # Map various business attributes to dictionaries for quick access
    bus_to_stars_map = business_rdd.map(lambda x: (x['business_id'], x['stars'])).collectAsMap()
    bus_is_open_map = business_rdd.map(lambda x: (x['business_id'], x['is_open'] == 1)).collectAsMap()
    bus_review_count = business_rdd.map(lambda x: (x['business_id'], x['review_count'])).collectAsMap()

    bus_is_restaurant_map = business_rdd.map(lambda x: (x['business_id'],
                                                        'restaurant' in x['categories'].lower() if 'categories' in x and
                                                                                                   x[
                                                                                                       'categories'] is not None else False)).collectAsMap()
    bus_is_shop_map = business_rdd.map(lambda x: (x['business_id'],
                                                  'shop' in x['categories'].lower() if 'categories' in x and
                                                                                       x[
                                                                                           'categories'] is not None else False)).collectAsMap()

    # Process and map specific restaurant attributes
    restaurants_rdd = business_rdd.filter(lambda x: bus_is_restaurant_map[x['business_id']]).cache()
    restaurant_attire_encodings = {'casual': 1, 'formal': 2, 'dressy': 3}
    restaurant_attire_map = restaurants_rdd.map(lambda x: (x['business_id'], restaurant_attire_encodings[
        x['attributes']['RestaurantsAttire']] if 'attributes' in x and x['attributes'] is not None and
                                                 'RestaurantsAttire' in x['attributes'] else 0)).collectAsMap()
    restaurant_price_map = restaurants_rdd.map(lambda x: (x['business_id'], int(
        x['attributes']['RestaurantsPriceRange2']) if 'attributes' in x and x['attributes'] is not None and
                                                      'RestaurantsPriceRange2' in x[
                                                          'attributes'] else 0)).collectAsMap()

    # Map user-specific attributes for restaurants and shops
    user_avg_restaurant_rating = train_rdd.filter(lambda x: bus_is_restaurant_map[x[1]]).map(
        lambda x: (x[0], (x[2], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(
        lambda x: x[0] / x[1]).collectAsMap()
    user_avg_shop_rating = train_rdd.filter(lambda x: bus_is_shop_map[x[1]]).map(
        lambda x: (x[0], (x[2], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).mapValues(
        lambda x: x[0] / x[1]).collectAsMap()

    # Map additional user attributes
    user_usefulness_map = user_rdd.map(lambda x: (x['user_id'], x['useful'])).collectAsMap()
    user_fans_map = user_rdd.map(lambda x: (x['user_id'], x['fans'])).collectAsMap()
    user_cool_map = user_rdd.map(lambda x: (x['user_id'], x['cool'])).collectAsMap()
    user_funny_map = user_rdd.map(lambda x: (x['user_id'], x['funny'])).collectAsMap()
    user_avg_rating_map = user_rdd.map(lambda x: (x['user_id'], x['average_stars'])).collectAsMap()
    user_total_reviews_count_map = user_rdd.map(lambda x: (x['user_id'], x['review_count'])).collectAsMap()

    # Calculate averages for user attributes
    avg_usefulness = user_rdd.map(lambda x: x['useful']).mean()
    avg_fans = user_rdd.map(lambda x: x['fans']).mean()
    avg_cool = user_rdd.map(lambda x: x['cool']).mean()
    avg_funny = user_rdd.map(lambda x: x['funny']).mean()
    avg_rating = user_rdd.map(lambda x: x['average_stars']).mean()

    # Average business stars
    avg_stars = business_rdd.map(lambda x: x['stars']).mean()

    # Define column mappings for DataFrame construction
    col_to_dict = {
        'user': [
            ('avg_user_rating', user_avg_rating_map, avg_rating),
            ('tot_user_reviews', user_total_reviews_count_map, 0),
            ('user_funny', user_funny_map, avg_funny),
            ('user_cool', user_cool_map, avg_cool),
            ('user_fans', user_fans_map, avg_fans),
            ('user_useful', user_usefulness_map, avg_usefulness)
        ],
        'business': [
            ('avg_bus_rating', bus_to_stars_map, avg_stars),
            ('tot_bus_reviews', bus_review_count, 0),
            ('bus_is_restaurant', bus_is_restaurant_map, False),
            ('restaurant_attire', restaurant_attire_map, 0),
            ('restaurant_price', restaurant_price_map, 0),
            ('bus_is_open', bus_is_open_map, True),
            ('review_count', bus_review_count, 0),
            ('bus_is_shop', bus_is_shop_map, False)
        ]
    }

    def prep_data(in_row, col_dict):
        """
        Prepares a single row of data for training or validation based on the mappings provided.

        Parameters:
        - in_row: The input row of data.
        - col_dict: Dictionary containing column mappings and default values.

        Returns:
        - List of processed data values.
        """
        user, business, true_rating = in_row
        ret_list = []

        # Append user-related features
        for col in col_dict['user']:
            default = col[2] if not isinstance(col[2], tuple) else col[2][0].get(user, col[2][1])
            ret_list.append(col[1].get(user, default))

        # Append business-related features
        for col in col_dict['business']:
            default = col[2] if not isinstance(col[2], tuple) else col[2][0].get(business, col[2][1])
            ret_list.append(col[1].get(business, default))

        ret_list.append(true_rating)
        return ret_list

    # Compile the attribute names
    cols = [c[0] for c in col_to_dict['user']] + [c[0] for c in col_to_dict['business']] + ['true_rating']
    attributes = cols[:-1]  # Exclude the target variable from attributes

    # Process the RDDs into DataFrames
    data_to_train = train_rdd.map(lambda x: prep_data(x, col_to_dict)).collect()
    train_df = pd.DataFrame(data_to_train, columns=cols)
    data_to_validate = validation_rdd.map(lambda x: prep_data(x, col_to_dict)).collect()
    validation_df = pd.DataFrame(data_to_validate, columns=cols)

    return train_df, validation_df, attributes
