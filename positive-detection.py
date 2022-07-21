'''
We have a graph where the nodes are tweets (they're actually words lol) and the edges indicate that the same author 
tweeted both tweets. We want to see whether the author is a positive person depending on the content of their tweets. 
We will use (or at least try to use) a GCN for this.
'''

import numpy as np
import gensim
from gensim.test.utils import datapath
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def average(feat_vectors):
    new_vec = 0
    for i in range(len(feat_vectors)):
        new_vec = new_vec + feat_vectors[i]
    
    return new_vec / len(feat_vectors)

def neural_net(feat_vector):
    # basically, we just have to apply weights, bias and activation function
    # since we would need to train a model to get the weights, we just simply return
    # the feature vector
    return feat_vector

def message_passing(feat_vectors, neighbors):
    # message passing:
    #   for each node
    #       aggregate all the neighbor feature vectors (lets say, average)
    #       put the feature vector through a neural net
    #       update the nodes feature vector
    # this allows the message to pass only to immediate neighbors and therefore, requires only one layer

    new_feat_vec = {}
    for node in range(num_nodes):
        f_v = []
        for i in neighbors[node]:
            f_v.append(feat_vectors[i])
        agg_feature_vector = average(f_v)
        new_feat_vec[node] = neural_net(agg_feature_vector)
    
    return new_feat_vec
    
def determine_stance(new_feat_vec):
    # to determine if the author is a positive person, average all the tweets of the person (I guess) and compare it with
    # the feature vectors of the word "positive" and "negative" (euclidean distance)
    pos_vec = model["positive"]
    avg_vec = average(new_feat_vec)

    pos_dist = np.linalg.norm(pos_vec - avg_vec)

    if pos_dist < 0.8: # estimated yet arbitrary values
        return "Positive"
    elif pos_dist > 1.5:
        return "negative"
    else:
        return "neutral"


# define a graph of tweets

# positive
'''
tweets = np.array([["good"],
                   ["great"],
                   ["like"],
                   ["happy"],
                   ["nice"]])
'''
'''
# negative
tweets = np.array([["no"],
                   ["attacks"],
                   ["down"],
                   ["crime"],
                   ["fire"]])
                   

'''

# neutral/negative
tweets = np.array([["good"],
                   ["great"],
                   ["no"],
                   ["crime"],
                   ["fire"]])


graph = np.array([[0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0],
                  [1, 1, 0, 1, 0],
                  [0, 1, 1, 0, 1],
                  [0, 0, 0, 1, 0]], dtype=int)

neighbors = np.array([[1, 2],
                 [0, 3, 4],
                 [0, 1, 3],
                 [1, 2, 4],
                 [3]])

num_nodes = len(neighbors)

# define a feature vector for each tweet
model = gensim.models.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)

feat_vectors = [model[t] for t in tweets]

# message passing
new_feat_vec = message_passing(feat_vectors, neighbors)

# determine the stance of the author of the tweets
print(f"In the 1-hop neighborhood, the author is a {determine_stance(new_feat_vec)} person")


# note: if we wanted to add another layer, then we just use the new_feat_vec as the feat_vector and carry on as usual.
# below is an example of how to add another layer
new_feat_vec = message_passing(new_feat_vec, neighbors)
print(f"In the 1-hop neighborhood, the author is a {determine_stance(new_feat_vec)} person")
