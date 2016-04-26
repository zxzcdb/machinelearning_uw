import graphlab as gl
import numpy as np
import sqrt from math

sales = gl.SFrame('kc_house_data.gl/')

def get_numpy_data(data, features, output):

    data['constant'] = 1

    features.append('constant')
    ft = data.select_columns(features)
    ftmatrix = ft.to_numpy()

    outarray = np.asarray(data[output])

    return(ftmatrix, outarray)

def predict_output(fm, wei):
    predictions = np.dot(fm, wei)
    return(predictions)

def feature_derivative(errors, feature):
    dev = np.dot(errors, feature)
    derivative = 2*dev
    return(derivative)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        initial_prediction = predict_output(feature_matrix,initial_weights)

        errors = initial_prediction - output

        gradient_sum_squares = 0

        for i in range(len(weights)):
            par_der = feature_derivative(errors,feature_matrix[:, i])
            gradient_sum_squares = gradient_sum_squares + par_der * par_der
            weights[i] = weights[i] + step_size * par_der

        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

# train and test
#Q1 & Q2
train_data,test_data = sales.random_split(.8,seed=0)

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

weights1 = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)

(feature_matrix1, output) = get_numpy_data(test_data, simple_features, my_output)
predict1 = predict_output(feature_matrix1,weights1)

#Q1
print "Q1: What is the value of the weight for sqft_living -- the second element of ‘simple_weights’ " + weights1[1]

#Q2
print "Q2: What is the predicted price for the 1st house in the Test data set for model 1" + predict1[0]

#Q3, Q4 & Q5
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

weights2 = regression_gradient_descent(model_features, output,initial_weights, step_size,tolerance)

(feature_matrix2, output) = get_numpy_data(test_data, simple_features, my_output)
predict2 = predict_output(model_features,weights2)

#Q3
print "Q3: What is the predicted price for the 1st house in the TEST data set for model 2 (round to nearest dollar)?" \
      + predict2[0]

d1 = output - predict1
d2 = output - predict2
#Q4
print "Q4: Which estimate was closer to the true price for the 1st house on the TEST data set, model 1 or model 2?" \
      + d1[0] + d2[0]

#Q5
print "Q5: Quiz Question: Which model (1 or 2) has lowest RSS on all of the TEST data? " + d1[0] + d2[0]
