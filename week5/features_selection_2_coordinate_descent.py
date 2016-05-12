import graphlab as gl
import numpy as np
import math
from regression import get_numpy_data
from regression import normalize_features
from regression import predict_output
from regression import lasso_coordinate_descent_step
from regression import lasso_cyclical_coordinate_descent
from regression import get_simple_residuals

sales = gl.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int)

print "==== To test normalize_features ==="

features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
print "Should print: \n[[ 0.6  0.6  0.6]\n  [ 0.8  0.8  0.8]]"# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
print "Should print: \n[5.  10.  15.]"
# should print
# [5.  10.  15.]

print "==== Implementing Coordinate Descent with normalized features ===="

print "=== Effect of L1 penalty ==="

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

weights = np.array([1., 4., 1.])

prediction = predict_output(simple_feature_matrix, weights)

ro = {}
for i in range(len(weights)):
    feature_i = simple_feature_matrix[:,i]
    tmp = feature_i * (output - prediction + weights[i]*feature_i)
    ro[i] = tmp.sum()
    print "ro[", i, "] is:", ro[i]
print "***** Quiz question *****"
print "Range 1 of L1 is [", 2*ro[2], ", ", 2*ro[1], ")."
print "Range 2 of L1 is lambda <", 2*ro[2]
#coordinate_decent(prediction, simple_feature_matrix, weights, )

print "Test function. Answer should print 0.425558846691."
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]),
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)

print "===== Cyclical coordinate descent ====="
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix,
                                            output, initial_weights, l1_penalty, tolerance)

rss = get_simple_residuals(predict_output(normalized_simple_feature_matrix, weights), output)
print "**** Quiz question ****"
print "RSS is: ", rss
print "**** Quiz question ****"
print "Which features had weight zero at convergence?", simple_features, weights

print "==== Evaluating LASSO fit with more features *****"
train_data,test_data = sales.random_split(.8,seed=0)
all_features = ['bedrooms',
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
                'yr_renovated']
op = 'price'
(all_feature_matrix, output) = get_numpy_data(sales, all_features, op)
(normalized_all_feature_matrix, all_norms) = normalize_features(all_feature_matrix)

l1_penalty = 1e7
initial_weights = np.zeros(14)
tolerance = 1
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print "**** Quiz question: What features had non-zero weight in this case?"
print "L1 = 1e7",all_features, weights1e7

l1_penalty= 1e8
initial_weights = np.zeros(14)
tolerance = 1
weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print "**** Quiz question: What features had non-zero weight in this case?"
print "L1 = 1e8", all_features, weights1e8

l1_penalty= 1e4
initial_weights = np.zeros(14)
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print "L1 = 1e4: ",all_features, weights1e4

print "==== Rescaling learned weights ===="

weights1e7_normalized = weights1e7 / all_norms
print "Normalized weights 1e7 is: ", weights1e7_normalized
print "Test: normalized_weights1e7[3] should return 161.31745624837794", weights1e7_normalized[3]

weights1e8_normalized = weights1e8 / all_norms
print "Normalized weights 1e8 is: ", weights1e8_normalized

weights1e4_normalized = weights1e7 / all_norms
print "Normalized weights 1e4 is: ", weights1e4_normalized

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

rss_1e7 = get_simple_residuals(predict_output(test_feature_matrix,weights1e7_normalized), test_output)
print "RSS of test data with L1 1e7 is: ", rss_1e7

rss_1e8 = get_simple_residuals(predict_output(test_feature_matrix,weights1e8_normalized), test_output)
print "RSS of test data with L1 1e8 is: ", rss_1e8

rss_1e4 = get_simple_residuals(predict_output(test_feature_matrix,weights1e4_normalized), test_output)
print "RSS of test data with L1 1e4 is: ", rss_1e4

print "*** Quiz question ***\nWhich model performed best on the test data?"
