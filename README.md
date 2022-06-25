# Basics-of-GNNs
Based on Zak Jost's course: https://www.graphneuralnets.com/p/basics-of-gnns

## water-drop:
  Description: We want to simulate message passing here. Suppose we have a graph where one node is "infected" with a colored water drop (rather than clear water, I suppose). The colored water drop is a message and we want to see how that message gets passed through a graph.
  
  Actual code: https://colab.research.google.com/github/zjost/blog_code/blob/master/gcn_numpy/message_passing.ipynb#scrollTo=C7v3DuvRFGMW


## fraud-detection:
  Description: We want to simulate label propagation. Here, we have accounts that are linked to credit cards and if we know that one account is a fraud account, then we can "stain" other accounts who are linked to the same credit card. This allows usto predict whether an account is a fraud account or not by determining the probability of both.
