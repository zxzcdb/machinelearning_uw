import graphlab
import json
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]
import string
import numpy as np
from classification import logistic_regression

products = graphlab.SFrame('amazon_baby_subset.gl/')

products['sentiment']

products.head(10)['name']

print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

print important_words

def remove_punctuation(text):
    return text.translate(None, string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

products['perfect']
products['contains_perfect'] = ""
ans1 = 0
for perfect in products['perfect']:
    if perfect >= 1:
        products['contains_perfect'] = 1
        ans1 = ans1 + 1
    else:
        products['contains_perfect'] = 0

print "Q2: How many reviews contain the word perfect?", ans1

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')

arrays = np.load('module-3-assignment-numpy-arrays.npz')
feature_matrix, sentiment = arrays['feature_matrix'], arrays['sentiment']

print "Q3: How many features: "

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)

scores = np.dot(feature_matrix, coefficients)

class_prediction = []
positive = 0
for score in scores:
    if score > 0:
        class_prediction.append(1)
        positive = positive + 1
    else:
        class_prediction.append(-1)

print "Q6: Positive reviews: ", positive

num_mistakes = 0
for i in range(len(products)):
    if products['sentiment'][i] <> class_prediction[i]:
        num_mistakes = num_mistakes + 1
accuracy = float(num_mistakes) / float(len(products))
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Q7: Accuracy = %.2f' % accuracy

coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

