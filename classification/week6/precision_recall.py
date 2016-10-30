import graphlab
import numpy as np

products = graphlab.SFrame('amazon_baby.gl/')

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

# Remove punctuation.
review_clean = products['review'].apply(remove_punctuation)

# Count words
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# Drop neutral sentiment reviews.
products = products[products['rating'] != 3]

# Positive sentiment to +1 and negative sentiment to -1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                            features=['word_count'],
                                            validation_set=None)

accuracy= model.evaluate(test_data, metric='accuracy')['accuracy']
print "Test Accuracy: %s" % accuracy

baseline = len(test_data[test_data['sentiment'] == 1])/float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline

confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']
confusion_matrix

precision = model.evaluate(test_data, metric='precision')['precision']
print "Precision on test data: %s" % precision

recall = model.evaluate(test_data, metric='recall')['recall']
print "Recall on test data: %s" % recall

def apply_threshold(probabilities, threshold):
    return probabilities.apply(lambda x: +1 if x >= threshold else -1)

probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

print "Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum()

print "Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum()

# Threshold = 0.5
precision_with_default_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_default_threshold)

recall_with_default_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_high_threshold)
recall_with_high_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_high_threshold)

print "Precision (threshold = 0.5): %s" % precision_with_default_threshold
print "Recall (threshold = 0.5)   : %s" % recall_with_default_threshold

print "Precision (threshold = 0.9): %s" % precision_with_high_threshold
print "Recall (threshold = 0.9)   : %s" % recall_with_high_threshold

print "=== Precision-recall curve ==="

threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values

precision_all = []
recall_all = []

probabilities = model.predict(test_data, output_type='probability')

for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    precision = graphlab.evaluation.precision(test_data['sentiment'], predictions)
    recall = graphlab.evaluation.recall(test_data['sentiment'], predictions)

    precision_all.append(precision)
    recall_all.append(recall)

    print "threshold: ", threshold
    print "precision: ", precision

predictions98 = apply_threshold(probabilities, 0.98)
print graphlab.evaluation.confusion_matrix(test_data['sentiment'],predictions98)

print "=== Evaluating specific search terms ==="

baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]

probabilities = model.predict(baby_reviews, output_type='probability')

threshold_values = np.linspace(0.5, 1, num=100)

precision_all = []
recall_all = []

for threshold in threshold_values:
    # Make predictions. Use the `apply_threshold` function
    predictions = apply_threshold(probabilities, threshold)
    precision = graphlab.evaluation.precision(baby_reviews['sentiment'], predictions)
    recall = graphlab.evaluation.precision(baby_reviews['sentiment'], predictions)

    # Append the precision and recall scores.
    precision_all.append(precision)
    recall_all.append(recall)
    print "threshold: ", threshold
    print "precision", precision
