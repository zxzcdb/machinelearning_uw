#  source ~/test27/bin/activate

import graphlab as gl
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from math import sqrt

# Week1: Simple Linear Regression
def simple_linear_regression(input, output):
    num = input.size()
    in_sum = input.sum()
    out_sum = output.sum()
    pro = input * output
    pro_sum = pro.sum()
    in_sq = input * input
    in_sq_sum = in_sq.sum()
    avi = input.mean()
    avo = output.mean()
    slope = (pro_sum - num*avi*avo)/(in_sq_sum-num*avi*avi)
    intercept = avo - avi*slope
    return (intercept, slope)
    
def lr_get_residual_sum_of_squares(input, output, intercept, slope):
    predict = input * slope + intercept

    residuals = predict - output
    rss = residuals*residuals
    RSS = rss.sum()

    return(RSS)
    
def get_regression_predictions(input, intercept, slope):
    predicted_values = input*slope + intercept
    return predicted_values

# Week2: Multiple Linear Regression: Assignment1
def get_residual_sum_of_squares(model, data, outcome):
    prediction = model.predict(data)
    residuals = prediction - outcome
    RSS = residuals*residuals
    RSS = RSS.sum()
    return(RSS)

# Week2: Multiple Linear Regression: Assignment2 Gradient Decent
def get_numpy_data(data, features, output):
    data['constant'] = 1
    features = ['constant'] + features
    ft = data.select_columns(features)
    ftmatrix = ft.to_numpy()
    outarray = np.asarray(data[output])
    return(ftmatrix, outarray)

def predict_output(fm, wei):
    predictions = np.dot(fm, wei)
    return(predictions)

def feature_derivative(errors, feature):
    der = np.dot(feature, errors)*2
    return(der)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        initial_prediction = predict_output(feature_matrix,initial_weights)
        errors = initial_prediction - output
        gradient_sum_squares = 0

        for i in range(len(weights)):
            deri = feature_derivative(errors,feature_matrix[:, i])
            gradient_sum_squares = gradient_sum_squares + (deri * deri)
            weights[i] = weights[i] - (step_size * deri)
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

# Week 3: Performance Assessment
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    if degree < 0:
        print("ERROR: degree should be >=0.")
        sys.exit(1)

    # initialize the SFrame:
    poly_sframe = gl.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature

    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)

    return poly_sframe

def polynomial_fit(data, degree):
    poldata = polynomial_sframe(data['sqft_living'],degree)
    feature = poldata.column_names()

    poldata['price'] = data['price']

    model = gl.linear_regression.create(poldata, target = 'price', features = feature, validation_set = None)
    model.get("coefficients")

    plt.plot(poldata['power_1'],poldata['price'],'.',poldata['power_1'], model.predict(poldata),'-')

# Week 4: Observe Overfit
def slice_data(n,k,i):
    # i starts with 0.
    if i <0:
        print "ERROR: i < 0!"
    start = (n*i)/k
    end = (n*(i+1))/k-1
    end += 1
    return (start, end)

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    poldata = polynomial_sframe(data.select_column(features_list[0]), 15)
    poldata[output_name] = data[output_name]
    for i in range(0,k):
        (start, end) = slice_data(len(data),k,i)
        train_data = poldata[0:start]
        train_data = train_data.append(poldata[end:len(data)])
        vali_data = data[start:end]
        model = gl.linear_regression.create(train_data, target='price',
                                            l2_penalty=l2_penalty,validation_set=None)
        rss = get_residual_sum_of_squares(model, vali_data, output_name)
        return rss

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100, verbose=False):

    weights = np.array(initial_weights)  # make sure it's a numpy array

    if verbose is True:
        print "initial weights: %s", str(weights)

    # while not reached maximum number of iterations:
    for iter in range(0, max_iterations):
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        test_predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = test_predictions - output

        for i in xrange(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, you are computing the derivative of the constant!)
            if i == 0:
                is_constant = True
            else:
                is_constant = False
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, is_constant)

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - derivative * step_size

        if verbose is True:
            print("iteration = %d, derivative = %f, weights = %s" % (iter, derivative, str(weights)))

    return weights
