# Basics-of-GNNs
Based on Zak Jost's course: https://www.graphneuralnets.com/p/basics-of-gnns

## water-drop:
  Description: We want to simulate message passing here. Suppose we have a graph where one node is "infected" with a colored water drop (rather than clear water, I suppose). The colored water drop is a message and we want to see how that message gets passed through a graph.
  
  Actual code: https://colab.research.google.com/github/zjost/blog_code/blob/master/gcn_numpy/message_passing.ipynb#scrollTo=C7v3DuvRFGMW


## fraud-detection:
  Description: We want to simulate label propagation. Here, we have accounts that are linked to credit cards and if we know that one account is a fraud account, then we can "stain" other accounts who are linked to the same credit card. This allows usto predict whether an account is a fraud account or not by determining the probability of both.

## positive-detection:
  Description: We have a graph where the nodes are tweets (they're actually words lol) and the edges indicate that the same author 
  tweeted both tweets. We want to see whether the author is a positive person depending on the content of their tweets. 
  We will use (or at least try to use) a GCN (Graph Convolution Network) for this.

## likeability:
  Description: We have a social network. We want to determine how likeable a person is. A very simple rule: if a person follows
  more people, they are likeable. If they block more people, they are not likeable. If a person follows more
  likeable people, then they are likeable. We want to use message passing to see how likeable their neighbors are. The results
  of this are quite interesting (assuming that the methodology is correct) and are included at the end of the Python file.

## attentative-positive-detection:
  Description: Exact same situation as the positive-detection situation. However, we have modified the situation so that we
  pay special attention to nodes depending on the dot product of a current node (the node gathering the messages)
  and the message node (the node from which we get the message). This is trying to simulate a Graph Attention Network.
