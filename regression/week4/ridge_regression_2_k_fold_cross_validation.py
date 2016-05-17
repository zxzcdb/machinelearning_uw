import graphlab as gl
import numpy as np
from regression import slice_data
from regression import k_fold_cross_validation
from regression import polynomial_sframe

print "*** Selecting an L2 penalty via cross-validation"

sales = gl.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living','price'])

# split the data set into training, validation and testing.
(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = gl.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

# i starts with 1
(start, end) = slice_data(n,k,3)

validation4 = train_valid_shuffled[start:end]

print "Test data slice. Answer should be 536234: ", int(round(validation4['price'].mean(), 0))

feature = polynomial_sframe(train_valid_shuffled['sqft_living'],15)

for l2 in np.logspace(1, 7, num=13):
    rss = k_fold_cross_validation(10, l2, train_valid_shuffled,'price',['sqft_living'])
    print "For L2_penalty is ", l2, ", validation error is", rss
