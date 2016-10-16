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

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    derivative = np.dot(errors, feature)
    if not feature_is_constant:
        derivative = derivative - 2 * l2_penalty * coefficient
    return derivative

def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)

    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)

    return lp

def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        ## YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            is_intercept = (j == 0)
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            ## YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j] = coefficients[j] + step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

def get_classification_accuracy(feature_matrix, sentiment, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    apply_threshold = np.vectorize(lambda x: 1. if x > 0  else -1.)
    predictions = apply_threshold(scores)

    num_correct = (predictions == sentiment).sum()
    accuracy = float(num_correct) / float(len(feature_matrix))
    return accuracy

def intermediate_node_num_mistakes(labels_in_node):

    safe_loans=0
    risky_loans=0
    # If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0

    # Count the number of 1's (safe loans)
    for i in np.nditer(labels_in_node):
        if(i==1):
            safe_loans = safe_loans + 1
        elif(i==-1):
            risky_loans = risky_loans + 1


    # Return the number of mistakes that the majority classifier makes.
    if(safe_loans > risky_loans):
        return risky_loans
    else:
        return safe_loans

def best_splitting_feature(data, features, target):

    best_feature = None # Keep track of the best feature
    best_error = 10     # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]

        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]

        # Calculate the number of misclassified examples in the left split.

        left_mistakes =  intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.

        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes+right_mistakes)/num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature=feature
            best_error=error

    return best_feature # Return the best feature

def create_leaf(target_values):

    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf':True   }

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1

    # Return the leaf node
    return leaf


def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):

    remaining_features = features[:]

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))


    # Stopping condition 1
    # (Check if there are mistakes at current node.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features ==[] :
        print "Stopping condition 2 reached."
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)

    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)


    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)

    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node **."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node ***."
        return create_leaf(right_split[target])

    # Repeat (recurse) on left and right subtrees
    print "Calling left tree",current_depth,splitting_feature
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)

    print "calling right tree",current_depth,splitting_feature
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)
    print "last return",splitting_feature
    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def classify(tree, x, annotate = False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            if annotate:
                print "Right Split ",tree['splitting_feature']
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data):
    # Apply classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))

    # Once you've made the prediction, calculate the classification error
    corrects = (prediction != data['safe_loans']) ## YOUR CODE HERE
#    print data['safe_loans']
#    print prediction
#    print corrects
    return corrects.sum() / float(len(data))
