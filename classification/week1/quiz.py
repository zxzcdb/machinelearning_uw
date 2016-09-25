import graphlab
import math
import string

products = graphlab.SFrame('amazon_baby.gl/')
products[269]

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

review_without_punctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_punctuation)

products[269]['word_count']


products = products[products['rating'] != 3]
len(products)

products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

print "=== Train a sentiment classifier with logistic regression ==="

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)

weights = sentiment_model.coefficients
weights.column_names()

num_positive_weights = weights[weights['value'] >= 0]
num_negative_weights = weights[weights['value'] < 0]

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

print "=== Making predictions with logistic regression ==="

sample_test_data = test_data[10:13]

print sample_test_data['rating']

sample_test_data

sample_test_data[0]['review']

sample_test_data[1]['review']

scores = sentiment_model.predict(sample_test_data, output_type='margin')

print scores

predict = []

for i in range(len(scores)):
    if scores[i] > 0:
        predict.append(1)
    else:
        predict.append(-1)

print "Class predictions are", predict

print "Class predictions according to GraphLab Create:"
print sentiment_model.predict(sample_test_data)

print "=== Probability predictions ==="

prob = []
for i in range(len(scores)):
    prob.append(1 / (1 + math.e**(-scores[i])))

print "Class predictions according to GraphLab Create:"
print sentiment_model.predict(sample_test_data, output_type='probability')

print "=== Find the most positive (and negative) review ==="

test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')

test_data.topk('probability',20).print_rows(20)

test_data.topk('probability',20, reverse = True).print_rows(20)

print "=== Compute accuracy of the classifier ==="

def get_classification_accuracy(model, data, true_labels):
    prediction = model.predict(data)

    correct_count = 0
    for i in range(len(data)):
        if prediction[i] == true_labels[i]:
            correct_count = correct_count + 1

    accuracy = float(correct_count) / float(len(data))
    return accuracy

get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])

print "=== Learn another classifier with fewer words ==="

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']

len(significant_words)

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)

train_data[0]['review']
print train_data[0]['word_count']
print train_data[0]['word_count_subset']

print "=== Train a logistic regression model on a subset of data ==="

simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
simple_model

get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

simple_model.coefficients

simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)

positive_significant_words = simple_model.coefficients['index'][simple_model.coefficients['value']>=0]

for word in positive_significant_words:
    print sentiment_model.coefficients['value'][sentiment_model.coefficients['index'] == word]

print "Accuracy of Sentiment_model:", get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
print "Accuracy of Simple model:", get_classification_accuracy(simple_model, train_data, train_data['sentiment'])

print "Accuracy of Sentiment_model:", get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
print "Accuracy of Simple model:", get_classification_accuracy(simple_model, test_data, test_data['sentiment'])

num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative

num_positive  = (test_data['sentiment'] == +1).sum()
num_negative = (test_data['sentiment'] == -1).sum()
print num_positive
print num_negative

print "Accuracy of Majority classifier is:", float(num_positive) / float(len(test_data))
