'''
Suppose we have a graph where one node is "infected" with a colored water drop (rather than clear water, I suppose).
The colored water drop is a message and we want to see how that message gets passed through a graph.
'''

import numpy as np

# define your graph using an adjacency matrix a
a = np.array([[0, 1, 0, 0, 0],
              [1, 0, 1, 0, 0],
              [0, 1, 0, 1, 1],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]])
num_nodes = len(a)
identity = np.eye(num_nodes)

# graph features are 0 except for one node that has a feature of 1 (this will be our colored water drop)
features = np.zeros((num_nodes, 1))
features[0, 0] = 1

# how do we get a-hat?
# a-hat = d^-1/2-tilde * a-tilde * d^-1/2-tilde
#       tilde means that you add the identity matrix to the original matrix
# d is the degree matrix

d = np.diag(np.sum(a, axis=1))
d_sqrt_tilde = np.linalg.inv(np.sqrt(d + identity))

a_tilde = a + identity

a_hat = d_sqrt_tilde @ a_tilde @ d_sqrt_tilde

# multiply features by a-hat to simulate the message passing
# multiple that by a-hat and then you get another round of message passing
# do this until convergence

print(features)

while True:
    new_features = a_hat @ features
    if np.array_equal(new_features, features):
        break
    features = new_features

print(features)

