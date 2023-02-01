
# This allows us to read the cities.json file. The format of the file
# is 'Javascript Object Notation' (JSON). It's a way of representing
# essential data structures, like lists and key-value pairs.

import json

# Our data will be in this format:
#    {
#        "city": "Beloit", 
#        "growth_from_2000_to_2013": "2.9%", 
#        "latitude": 42.5083482, 
#        "longitude": -89.03177649999999, 
#        "population": "36888", 
#        "rank": "999", 
#        "state": "Wisconsin"
#    },

cities_dataset = json.load(open('cities.json'))

def manhattan_distance(starting_point, destination):
    """
    This is a function that tells you basically, 'If my coordinates were city blocks, and
    I couldn't walk through the middle of the block, how far would I have to walk to get from
    a to b, using city streets?
    """
    distance = 0
    n_dimensions = len(starting_point)
    for dimension in range(n_dimensions):
        # Could be streets, could be avenues, could be anything!
        distance_at_dimension = abs(starting_point[dimension] - destination[dimension])
        distance += distance_at_dimension
    return distance

def normalized_dot_product(a, b):
    """
    This is a function that tells you basically, 'how much are two vectors pointing in the same direction?'
    This is just the Pythagorean theorem!
    """
    # normalize vectors.
    # A ** 2 + B ** 2 = C ** 2
    # Raise to power of 1/2 to get the square root
    a_length = sum([x**2 for x in a]) ** 0.5
    a_normalized = [x / a_length for x in a]

    b_length = sum([x**2 for x in b]) ** 0.5
    b_normalized = [x / b_length for x in b]

    # calculate dot product
    return sum([a_normalized[i] * b_normalized[i] for i in range(len(a))])


#if __name__ == '__main__':
    # Run this code if we're running this file directly
    # (as opposed to importing it from another file)

# Let's compare cities!

# First, we know that cities close together have some things in common.
# It's fair to say that Chicago and Minneapolis are 'similar' in some
# important ways. Let's see what computing the overlap of latitudes and
# longitudes looks like.

# We're going to store the cities in a dictionary
# The key will be the city name
# The value will be the coordinates of the city. Latitude and longitude
# are just a geographic, 2 dimensional vector:

lonlat = {}

for city in cities_dataset:
    lonlat[city['city']] = [city['longitude'], city['latitude']]

# Woah! That's a lot of cities! Let's simplify with ones with a bigger population:
big_cities = {}
for city in cities_dataset:
    if int(city['population']) > 100000:
        big_cities[city['city']] = [city['longitude'], city['latitude']]

# Now, let's make a function that finds the nearest city to each other city,
# using the oversimplified-but-fine approximation of the manhattan distance:
def approximate_nearest_city(city_name, cities):
    coordinates = cities[city_name]
    # So far, we have nothing 'nearest', so start our search with a large value:
    nearest = 1000
    nearest_name = None
    for other_city, other_coordinates in cities.items():
        # calculate the distance between the two cities
        distance = manhattan_distance(coordinates, other_coordinates)
        print('distance from', city, 'to', other_city, 'is', distance)
        # if the distance is less than the current nearest, update the nearest
        # unless the distance is near 0, which means we're comparing the same city
        # to itself
        if distance < nearest and distance > 0.000001:
            nearest = distance
            nearest_name = other_city

    # return the nearest city
    return nearest_name, distance

# Now, let's test it out. Also, we are going to see how long it takes to find
# the nearest city to each other city:
import time
starting_time = time.time()
print('Now find the city nearest to each other city:')
for city in big_cities:
    print(city, approximate_nearest_city(city, big_cities))

ending_time = time.time()

print('It took', ending_time - starting_time, 'seconds to find the nearest city to each other city.')
print('the nearest city to Chicago is', approximate_nearest_city('Chicago', big_cities))

# So this kind of works, but we want something a little deeper. More 'similarity' than just
# 'nearness'. Let's try to find the most similar city to each other city, using additional features.
# Intuitively, nearby cities do share a lot in common. But also, big cities have some of that bigness
# in common, apart from geography. Let's see what we can do with that.

# Now, let's start thinking of our cities list as a matrix. Each row is a city, and each column is a feature.
big_cities_list = [city for city in cities_dataset if int(city['population']) > 100000]
names = [city['city'] for city in big_cities_list]
longitudes = [city['longitude'] for city in big_cities_list]
latitudes = [city['latitude'] for city in big_cities_list]
populations = [float(city['population']) for city in big_cities_list]
growth_rates = [float(city['growth_from_2000_to_2013'].replace('%', '')) for city in big_cities_list]

# So far, so good, but how do we compare the number of people in one city to the degrees latitude of another?
# Answer: you don't! Intuitively, we know that longitude, the Eastness-Westness of a city, does mean something
# to us. Seattle has some things in common with San Francisco much further South. And likewise Phoenix to
# Jacksonville in terms of latitude -- there's a real "sunbelt" effect uniting these distant cities. So we want
# to keep all those features, but not over-emphasize any in particular. Again, we're designing our features here.
# And BTW, since we're designing features, let's throw in one more: the market share of pickup trucks in the state!
# Because why not?! Also lets us practice reading CSV files.

# our format will be:
# Connecticut,9.4%
# New Jersey,7.8%
# ...

import csv
share = csv.DictReader(open('pickup-trucks.csv'), delimiter = ',', fieldnames=['state', 'market_share'])
state_truck_share = {}
for state in share:
    state_name = state['state']
    market_share = float(state['market_share'].replace('%', ''))
    state_truck_share[state_name] = market_share

pickup_market_share = [state_truck_share[city['state']] for city in big_cities_list]

# Back to our matrix! We definitely won't be comparing longitudes to pickups! So let's rescale all our features

# We'll use this common function:

def rescale(vector, range = (-0.5, 0.5)):
    """
    This function takes a vector of numbers, and rescales it to be between
    the range specified. The default range is -0.5 to 0.5, which is where
    floating point numbers are most accurate.
    """
    # find the minimum and maximum values in the vector
    min_value = min(vector)
    max_value = max(vector)
    # find the range of the vector
    vector_range = max_value - min_value
    # find the range we want to rescale to
    target_range = range[1] - range[0]
    # rescale the vector
    rescaled = [range[0] + (x - min_value) * target_range / vector_range for x in vector]
    return rescaled

# Now, let's rescale all our features:
rescaled_longitudes = rescale(longitudes)
rescaled_latitudes = rescale(latitudes)
rescaled_populations = rescale(populations)
rescaled_growth_rates = rescale(growth_rates)
rescaled_pickup_market_share = rescale(pickup_market_share)

# Now, let's put all our features together into a matrix:
import numpy as np

matrix = np.array([rescaled_longitudes, rescaled_latitudes, rescaled_populations,
    rescaled_growth_rates, rescaled_pickup_market_share])

# Now, let's transpose the matrix, so that each row is a city, and each column is a feature:

matrix = matrix.T

# Now, let's take a couple of approaches. Here we normalize so that each city's vector has length 1:
normalized_matrix = matrix / np.linalg.norm(matrix, axis = 1).reshape(-1, 1)

# Working with the normalized matrix is really like "how much are these two vectors pointing in the same direction?"
# Because the magnitude of each row is 1, we can just take the dot product of each row with every other row. This
# is the same as matrix multiplication, but we're multiplying a matrix by its transpose. So we can use the @ operator

similarities = normalized_matrix @ normalized_matrix.T

# Now, arrange the results such that the most similar city is at the top of the list (which will be itself), and
# the least similar city is at the bottom of the list (which will be the city furthest away in terms of our combined
# features):

# Axis = 1 means we're sorting each row, and -1 means we're sorting in descending order
# Argsort returns the indices of the sorted array -- where you would have to put stuff so that it appears in order
rankings = np.argsort(-similarities, axis = 1)

# Now, let's print out the most similar city to each other city:
print('Now find the most similar city to each other city:')

def show_rankings(rankings, names, n=5, preamble=''):
    for index, name in enumerate(names):
        top_n = rankings[index][1:1+n]
        near = [names[i] for i in top_n]
        far = [names[i] for i in rankings[index][-n:]]
        print(preamble, name, 'is most similar to', ', '.join(near), 'and least similar to', ', '.join(far))

show_rankings(rankings, names)

# Now, just for fun, let's ignore geography and just look at the features.
no_geo = matrix[:, 2:]
# Re-normalize:
no_geo = no_geo / np.linalg.norm(no_geo, axis = 1).reshape(-1, 1)

# Let's talk about distances, without geography.
no_geo_distances = no_geo @ no_geo.T
no_geo_rankings = np.argsort(-no_geo_distances, axis = 1)

print('Now find the most similar city to each other city, ignoring geography:')
show_rankings(no_geo_rankings, names)


# Time for deep learning! We see above a couple of things:
# 1. Geography is either too dominant or not dominant enough.
# 2. We need more to capture the subtlety of underlying features of these cities.

# What if instead of making up a feature vector for each city, we let loose a neural network,
# with the broad instructions of: "find the features that best capture the semantic essence
# of every word and image on the internet"? Let's see what happens!

# First, we need to get our data into a format that a neural network can understand.
# How about English!

import sentence_transformers
model = sentence_transformers.SentenceTransformer('clip-ViT-B-32').to('cuda')

def embed_row(row):
    # Little bit of practical engineering here: language models are not exceptionally
    # good at dealing with numbers. And the really big ones don't entirely need to be!
    # So let's just use the proper name for the city an state, and see what happens.
    sentence = f"""{row.get('city')}, {row['state']}"""
    print('getting embedding for', sentence)
    return model.encode(sentence)

# Now, let's embed all our cities:
embeddings = np.array([embed_row(row) for row in big_cities_list])

# Now, let's see how similar these embeddings are to each other:
embeddings_distances = embeddings @ embeddings.T
embeddings_rankings = np.argsort(-embeddings_distances, axis = 1)


def top_n_ranking(rankings, index, n=5):
    top_n = rankings[index][1:1+n]
    near = [names[i] for i in top_n]
    far = [names[i] for i in rankings[index][-n:]]
    return near, far


def show_ranking(rankings, index, preamble=''):
    near, far = top_n_ranking(rankings, index)
    print(preamble, 'is most similar to:', ', '.join(near))
    print(preamble, 'is least similar to:', ', '.join(far))


def compare_models(city_name, state_name):
    # Almost 10 of America's top 300 cities share the same name. So we need to be more specific.
    idx = 0
    found = False
    for row in big_cities_list:
        if row['city'] == city_name and row['state'] == state_name:
            found = True
            break
        idx += 1
    if not found:
        print("Couldn't find", city_name, 'in', state_name)
        return
    print('Now find the most similar city to', city_name, 'in each model:')
    show_ranking(rankings, idx, preamble=f'With our manually created basic features, {city_name}, {state_name} ')
    show_ranking(embeddings_rankings, idx, preamble=f'With our deep learning model, {city_name}, {state_name} ')