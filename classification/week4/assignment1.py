import graphlab
from classification import intermediate_node_num_mistakes
from classification import best_splitting_feature
from classification import create_leaf
from classification import intermediate_node_num_mistakes
from classification import classify
from classification import evaluate_classification_error

loans = graphlab.SFrame('lending-club-data.gl/')

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

print "=== Subsample dataset to make sure classes are balanced ==="

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

print "=== Transform categorical data into binary features ==="

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

print "=== Train-Validation split ==="

train_data, validation_set = loans_data.random_split(.8, seed=1)

def reached_minimum_node_size(data, min_node_size):
    if len(data) <= min_node_size:
            return True

def error_reduction(error_before_split, error_after_split):
    return(error_before_split - error_after_split)

def decision_tree_create(data, features, target, current_depth = 0,
                         max_depth = 10, min_node_size=1,
                         min_error_reduction=0.0):

    remaining_features = features[:] # Make a copy of the features.

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))


    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size) == True:
        print "Early stopping condition 2 reached. Reached minimum node size."
        return reached_minimum_node_size(data, min_node_size)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes =  intermediate_node_num_mistakes(left_split[target])
    right_mistakes =  intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if  error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Early stopping condition 3 reached. Minimum error reduction."
        return  create_leaf(target_values)


    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))


    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)

    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target,
                                      current_depth + 1, max_depth, min_node_size, min_error_reduction)


    return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2,
                                        min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 7'

my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 100, min_error_reduction=0.0)

my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)

validation_set[0]

print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_set[0])

classify(my_decision_tree_new, validation_set[0], annotate = True)

classify(my_decision_tree_old, validation_set[0], annotate = True)

evaluate_classification_error(my_decision_tree_new, validation_set)

model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2,
                                min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 14,
                                min_node_size = 0, min_error_reduction=-1)

# model_1: max_depth = 2 (too small)
# model_2: max_depth = 6 (just right)
# model_3: max_depth = 14 (may be too large)
print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data)

# model_1: max_depth = 2 (too small)
# model_2: max_depth = 6 (just right)
# model_3: max_depth = 14 (may be too large)
print "Training data, classification error (model 1):", evaluate_classification_error(model_1, validation_set)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, validation_set)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, validation_set)

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

# model_1: max_depth = 2 (too small)
# model_2: max_depth = 6 (just right)
# model_3: max_depth = 14 (may be too large)
print count_leaves(model_1)
print count_leaves(model_2)
print count_leaves(model_3)

# model_1: max_depth = 2 (too small)
# model_2: max_depth = 6 (just right)
# model_3: max_depth = 14 (may be too large)
print count_leaves(model_1)
print count_leaves(model_2)
print count_leaves(model_3)

# model_4: min_error_reduction = -1 (ignoring this early stopping condition)
# model_5: min_error_reduction = 0 (just right)
# model_6: min_error_reduction = 5 (too positive)
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 0, min_error_reduction=0)
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6,
                                min_node_size = 0, min_error_reduction=5)

# model_4: min_error_reduction = -1 (ignoring this early stopping condition)
# model_5: min_error_reduction = 0 (just right)
# model_6: min_error_reduction = 5 (too positive)
print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set)

# model_4: min_error_reduction = -1 (ignoring this early stopping condition)
# model_5: min_error_reduction = 0 (just right)
# model_6: min_error_reduction = 5 (too positive)
print count_leaves(model_4)
print count_leaves(model_5)
print count_leaves(model_6)

# model_7: min_node_size = 0 (too small)
# model_8: min_node_size = 2000 (just right)
# model_9: min_node_size = 50000 (too large)
model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                min_node_size=0, min_error_reduction=-1)
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                min_node_size=2000, min_error_reduction=-1)
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                min_node_size=50000, min_error_reduction=-1)

# model_7: min_node_size = 0 (too small)
# model_8: min_node_size = 2000 (just right)
# model_9: min_node_size = 50000 (too large)
print "Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_set)
print "Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_set)
print "Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_set)

# model_7: min_node_size = 0 (too small)
# model_8: min_node_size = 2000 (just right)
# model_9: min_node_size = 50000 (too large)
print count_leaves(model_7)
print count_leaves(model_8)
print count_leaves(model_9)

