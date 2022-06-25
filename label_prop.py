'''
We want to simulate label propagation. Here, we have accounts that are linked to credit cards and if we know that one
account is a fraud account then we can "stain" other accounts who are linked to the same credit card.
'''
import numpy as np
# f(t+1) = alpha * S * f(t) + (1-alpha) * y
#   f(t) are the predictions at time t (column 1 is for not fraud and column 2 is for fraud)
#       for example: [[0.2, 0.8],
#                      [0.7, 0.3]] means that the first account is most likely fraud while the second is not
#   S is the matrix of which accounts are connected to which credit cards
#   alpha is a hyper parameter
#   y are the initial labels for the nodes (fraud nodes have labels 0, 1 while not fraud have labels 1, 0)
#       any attribute nodes or unknown nodes will have labels 0.5, 0.5

# This is the graph:
# account 1 (fraud)
#           \
#            credit card 1
#           /
# account 2
#           \
#            credit card 2
#           /
# account 3 (not fraud)

num_accounts = 3

S_nonnorm = np.array([[0, 0, 0, 1, 0], 
                      [0, 0, 0, 1, 1], 
                      [0, 0, 0, 0, 1], 
                      [1, 1, 0, 0, 0], 
                      [0, 1, 1, 0, 0]])
alpha = 0.6

# account 1 is fraud, account 2 is unknown, account 3 is not fraud and credit cards 1/2 are attributes
y = np.array([[0, 1], [0.5, 0.5], [1, 0], [0.5, 0.5], [0.5, 0.5]])

#take into account the neighborhood size so we can get probabilities of an account being fraud or not
row_sums = S_nonnorm.sum(axis=1)
S = S_nonnorm / row_sums[:, np.newaxis]

f_old = y

while True:
    f_new = alpha * (S @ f_old) + (1 - alpha) * y
    if np.array_equal(f_new, f_old):
        break
    f_old = f_new

for i in range(num_accounts):
    if f_new[i, 0] > f_new[i, 1]:
        pred = "not fraud"
    elif f_new[i, 0] == f_new[i, 1]:
        pred = "unknown"
    else:
        pred = "fraud"
    print(f'Account {i+1} is {pred}')

