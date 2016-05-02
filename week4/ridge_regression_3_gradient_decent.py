import graphlab as gl
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from regression import get_numpy_data
from regression import predict_output
from regression import feature_derivative_ridge
from regression import ridge_regression_gradient_descent
from regression import get_simple_residuals

mpl.use('TkAgg')

sales = gl.SFrame('kc_house_data.gl/')

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''
# -5.65541667824e+13
# -5.65541667824e+13

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.
# -22446749336.0
# -22446749336.0

# Gradient Decent

simple_features = ['sqft_living']
my_output = 'price'

train_data,test_data = sales.random_split(.8,seed=0)

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

l2_penalty = 0

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                             output,
                                                             initial_weights,
                                                             step_size,
                                                             l2_penalty,
                                                             max_iterations)

print "Weights of model with 0 penalty are:", simple_weights_0_penalty

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,
                                                                output,
                                                                initial_weights,
                                                                step_size,
                                                                l2_penalty,
                                                                max_iterations)
print "Weights of model with high penalty are: ", simple_weights_high_penalty

plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')

test_data['price_1'] = predict_output(simple_test_feature_matrix, initial_weights)
test_data['price_2'] = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
test_data['price_3'] = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)

print 'Model of 1 feature:'
print "RSS of initial weights:             ", get_simple_residuals(test_data['price'], test_data['price_1'])
print "RSS of simple_weights_0_penalty:    ", get_simple_residuals(test_data['price'], test_data['price_2'])
print "RSS of simple_weights_high_penalty: ", get_simple_residuals(test_data['price'], test_data['price_3'])

# -------------------------------
# Running a multiple regression with L2 penalty
# -------------------------------
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0.
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                               output,
                                                               initial_weights,
                                                               step_size,
                                                               l2_penalty,
                                                               max_iterations)
print "multiple_weights_0_penalty is: ", multiple_weights_0_penalty

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix,
                                                                  output,
                                                                  initial_weights,
                                                                  step_size,
                                                                  l2_penalty,
                                                                  max_iterations)
print "Multiple_weights_high_penalty is: ", multiple_weights_high_penalty

test_data['price_1'] = predict_output(test_feature_matrix, initial_weights)
test_data['price_2'] = predict_output(test_feature_matrix, multiple_weights_0_penalty)
test_data['price_3'] = predict_output(test_feature_matrix, multiple_weights_high_penalty)

print 'Model of multiple features:'
print "RSS of initial weights:               ", get_simple_residuals(test_data['price'], test_data['price_1'])
print "RSS of multiple_weights_0_penalty:    ", get_simple_residuals(test_data['price'], test_data['price_2'])
print "RSS of multiple_weights_high_penalty: ", get_simple_residuals(test_data['price'], test_data['price_3'])
