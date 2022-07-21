'''
We have a graph where the nodes are tweets and the edges indicate that the same author tweeted both tweets. We 
want to see whether the author is a positive person depending on the content of their tweets. We will use (or at
least try to use) a GCN for this.
'''

# define a graph of tweets

# define a feature vector for each tweet

# message passing:
#   for each node
#       aggregate all the neighbor feature vectors (let's say, average)
#       put the feature vector through a neural net (apply weights, bias and activation function)
#       update the node's feature vector

# to determine if the author is a positive person, average all the tweets of the person (I guess)