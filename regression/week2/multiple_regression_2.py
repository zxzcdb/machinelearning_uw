#  source ~/test27/bin/activate

import graphlab as gl
import numpy as np
from regression import get_numpy_data
from regression import predict_output
from regression import regression_gradient_descent

sales = gl.SFrame('kc_house_data.gl/')

# train and test

#Q1 & Q2
train_data,test_data = sales.random_split(.8,seed=0)
simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

weights1 = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)

(feature_matrix1, output) = get_numpy_data(test_data, simple_features, my_output)
predict1 = predict_output(feature_matrix1,weights1)

#Q1
print "Q1: What is the value of the weight for sqft_living", weights1[1]

#Q2
print "Q2: What is the predicted price for the 1st house in the Test data set for model 1", predict1[0]

#Q3, Q4 & Q5
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

weights2 = regression_gradient_descent(model_features, output, initial_weights, step_size, tolerance)

(feature_matrix2, output) = get_numpy_data(test_data, simple_features, my_output)
predict2 = predict_output(model_features,weights2)

#Q3
print "Q3: What is the predicted price for the 1st house in the TEST data set for model 2?", predict2[0]

d1 = output - predict1
d2 = output - predict2
#Q4
print "Q4: Which estimate was closer to the true price for the 1st house on the TEST data set?", d1[0], d2[0]

#Q5
print "Q5: Quiz Question: Which model (1 or 2) has lowest deri on all of the TEST data? ", d1[0], d2[0]
