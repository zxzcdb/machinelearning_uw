import string
import math
import graphlab as gl
import numpy as np
from math import sqrt

def remove_punctuation(text):
    return text.translate(None, string.punctuation)

def get_classification_accuracy(model, data, true_labels):
    prediction = model.predict(data)

    correct_count = 0
    for i in range(len(data)):
        if prediction[i] == true_labels[i]:
            correct_count = correct_count + 1

    accuracy = float(correct_count) / float(len(data))
    return accuracy

def logistic_possibility(x,w):
    pos = 1 / (1 + math.e**(-x*w))
    return pos

def derivative(x,y,p):
    if y == 1:
        y = 1
    else:
        y = 0
    dev = x * (y - p)
    return dev

def total_derivative(xx,yy,w):
    total_deri = 0
    for i in range(len(xx)):
        pbt = logistic_possibility(xx[i],w)
        total_deri = total_deri + derivative(xx[i],yy[i],pbt)
    return total_deri

def predict_probability(feature_matrix, coefficients):
    product = np.dot(feature_matrix, coefficients)
    predictions = 1/(1 + math.e**(-product))
    return predictions

def feature_derivative(errors, feature):
    derivative = np.dot(errors, feature)
    return derivative

def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))

    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]

    lp = np.sum((indicator-1)*scores - logexp)
    return lp

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        predictions = predict_probability(feature_matrix, coefficients)

        indicator = (sentiment==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient

            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[:,j])
            coefficients[j] = coefficients[j] + step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
