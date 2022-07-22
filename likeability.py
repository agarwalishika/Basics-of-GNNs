'''
We have a social network. We want to determine how likeable a person is. A very simple rule: if a person follows
more people, they are likeable. If they block more people, they are not likeable. If a person follows more
likeable people, then they are likeable. We want to use message passing to see how likeable their neighbors are.
'''

from hashlib import new
import numpy as np
import gensim
from gensim.test.utils import datapath
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def average(feat_vectors):
    new_vec = np.zeros(len(feat_vectors[0]))
    for f_v in feat_vectors:
        new_vec = new_vec + f_v
    
    return new_vec / len(feat_vectors)

def neural_net(feat_vector, relation):
    # basically, we just have to apply weights, bias and activation function
    # since we would need to train a model to get the weights, we just simply return
    # the feature vector
    if relation == "B" and len(feat_vector) > 0:
        for i in range(len(feat_vector[0])):
            feat_vector[0][i] = -1 * feat_vector[0][i]
    return feat_vector

def message_passing(feat_vectors, neighbors, graph):
    # message passing:
    #   for each node
    #       aggregate all the neighbor feature vectors (lets say, average)
    #       put the feature vector through a neural net
    #       update the nodes feature vector
    # this allows the message to pass only to immediate neighbors and therefore, requires only one layer

    new_feat_vec = {}
    for node in range(num_nodes):
        follow_f_v = []
        block_f_v = []
        for i in neighbors[node]:
            if graph[node][i] == 1:
                follow_f_v.append(feat_vectors[i])
            elif graph[node][i] == -1:
                block_f_v.append(feat_vectors[i])
        follow_agg = neural_net(follow_f_v, "F")
        block_agg = neural_net(block_f_v, "B")
        new_feat_vec[node] = average(follow_agg + block_agg)
        
    
    return new_feat_vec
    
def determine_stance(new_feat_vec):
    # to determine if someone is likeable, add the follow-score and subtract the block score (I think)
    for i in new_feat_vec:
        if new_feat_vec[i][0] - new_feat_vec[i][1] > 0:
            print(f"Person {i} is likeable")
        else:
            print(f"Person {i} is not likeable")


# define a social network
# 0 = no connection, 1 = follows, -1 = blocks
graph = np.array([[0, 1, 1, 0, 0],
                  [1, 0, -1, 1, 0],
                  [1, -1, 0, 1, 0],
                  [0, 1, 1, 0, -1],
                  [0, 0, 0, -1, 0]], dtype=int)

neighbors = np.array([[1, 2],
                 [0, 2, 3],
                 [0, 1, 3],
                 [1, 2, 4],
                 [3]])

num_nodes = len(neighbors)

# define a feature vector for each tweet - number of people they follow / block / no connection

feat_vectors = []
for node in graph:
    node = list(node)
    feat_vectors.append([node.count(1), node.count(-1)])

# message passing
new_feat_vec = message_passing(feat_vectors, neighbors, graph)

# determine the stance of the author of the tweets
print("In the 1-hop neighborhood, here are the outcomes:")
determine_stance(new_feat_vec)


# note: if we wanted to add another layer, then we just use the new_feat_vec as the feat_vector and carry on as usual.
# below is an example of how to add another layer
new_feat_vec = message_passing(new_feat_vec, neighbors, graph)
print("In the 2-hop neighborhood, here are the outcomes:")
determine_stance(new_feat_vec)

'''
In the 1-hop neighborhood, here are the outcomes:
Person 0 is likeable
Person 1 is likeable
Person 2 is likeable
Person 3 is not likeable
Person 4 is not likeable
In the 2-hop neighborhood, here are the outcomes:
Person 0 is likeable
Person 1 is likeable
Person 2 is likeable
Person 3 is not likeable
Person 4 is likeable

Here are the results with the current configuration. This is quite interesting. Person 3/4 are blocked by each other. The only relation that
Person 4 has is the block relationship with Person 3. That's why Person 4 is labeled as "not likeable". However, Person 3 (who is connected
to Person 4) is connected to likeable people, which passes along to Person 4 and therefore, becomes likeable. Cool!
'''
