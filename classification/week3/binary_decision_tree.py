import graphlab
import numpy as np

loans = graphlab.SFrame('lending-club-data.gl/')

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features

print "Number of features (after binarizing categorical variables) = %s" % len(features)

loans_data['grade.A']

print "Total number of grade.A loans : %s" % loans_data['grade.A'].sum()
print "Expexted answer               : 6422"

train_data, test_data = loans_data.random_split(.8, seed=1)

print "=== Decision tree implementation ==="

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

print "=== Function to pick best feature to split on ==="

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

print "=== Building the tree ==="

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

print "=== Build the tree ==="

my_decision_tree = decision_tree_create(train_data, features, target, current_depth = 0, max_depth = 6)

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

def evaluate_classification_error(tree, data, target):
    # Apply the classify(tree, x) to each row in your data

    #print classify(tree,data)
    prediction = data.apply(lambda x:  classify(tree,x))

    # Once  made the predictions, calculate the classification error and return it
    mistakes=(prediction!=data[target]).sum()

    return (mistakes/float(len(data)))

print "Evaluate error:" evaluate_classification_error(my_decision_tree, test_data, target)

print "=== Printing out a decision stump ==="

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

print_stump(my_decision_tree)

print "=== Exploring the intermediate left subtree ==="

print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])
