import graphlab as gl
import numpy as np
from math import sqrt
from math import log

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
    if degree < 1:
        print "ERROR: degree < 1."
    poly_sframe = gl.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe

# Week 4: Ridge Regression
def poly_regression(data, l2):
    model = gl.linear_regression.create(data, target = 'price',
                                           l2_penalty = l2, validation_set = None)
    return(model)

def model_co(model):
    print(model.get("coefficients"))
