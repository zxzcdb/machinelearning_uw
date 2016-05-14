import graphlab as gl
import numpy as np
from regression import normalize_features
from regression import get_numpy_data
from regression import compute_distances
from regression import compute_knn_index
from regression import predict_knn
from regression import multiple_predict_knn
from regression import get_simple_residuals
import matplotlib.pyplot as plt

sales = gl.SFrame('kc_house_data_small.gl/')
(train_and_validation, test) = sales.random_split(.8, seed=1) # initial train/test split
(train, validation) = train_and_validation.random_split(.8, seed=1) # split training set into training and validation sets

feature_list = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

query = features_test[0]
print "1st row of teat features: ", query
comp = features_train[9]
print "10th row of training features: ", comp

distance = np.sqrt(np.sum((query-comp)**2))
print "*** QUIZ QUESTION ***"
print "Euclidean distance between the query house and the 10th house of the training set: ", distance

def compute_distance(query, compare):
    distance = np.sqrt(np.sum((query-compare)**2))
    return distance

dist_v = {}
print "*** QUIZ QUESTION ***"
for i in range(10):
    dist_v[i] = compute_distance(features_test[0], features_train[i])
    print "Distance to house[", i, "] is: ", dist_v[i]

diff = features_train[:] - features_test[0]
print "Should print -0.0934339605842:", diff[-1].sum()

print "Should be same: \n", np.sum(diff**2, axis=1)[15] # take sum of squares across each row, and print the 16th sum
print np.sum(diff[15]**2) # print the sum of squares for the 16th row -- should be same as above

distances = np.sqrt(np.sum((features_train[:] - features_test[0])**2,axis=1))

print "Should print 0.0237082324496:", distances[100] # Euclidean distance between the query house and the 101th training house
# should print 0.0237082324496

dist = compute_distances(features_test[2], features_train)
print "*** QUIZ QUESTIONS ***"
print "What is the index?:", np.argmin(dist)
print "What is the predicted value?", output_train[np.argmin(dist)]

print "=== Perform k-nearest neighbor regression ==="

print "*** QUIZ QUESTION ***"
print "Indices of the 4 training houses: ", compute_knn_index(4, features_test[2], features_train)
print "Predict value by knn is: ", predict_knn(4, features_test[2], features_train, output_train)[0]

print "=== Make multiple predictions ==="

print "*** QUIZ QUESTION"
ers = multiple_predict_knn(10, features_test[0:9],features_train,output_train)[0]
ind = multiple_predict_knn(10, features_test[0:9],features_train,output_train)[1]
min_index = np.argmin(ers)
print "What is the index: ", ind[min_index]
print "What is the predicted: ", min(ers)

print "=== Choosing the best value of k using a validation set ==="

kvals = range(1, 16)
rss_all = []
for k in kvals:
    print "For k=", k
    prediction = multiple_predict_knn(k, features_valid, features_train, output_train)[0]
    rss = get_simple_residuals(prediction, output_valid)
    rss_all.append(rss)
    print "RSS is: ", rss

print "*** QUIZ QUESTION"
print "What is the RSS: ", rss_all
plt.plot(kvals, rss_all,'bo-')
