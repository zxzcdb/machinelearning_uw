# source ~/test27/bin/activate
# pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/1.9/klhscott@163.com/FCEF-BFF5-FBDF-57DD-620D-9B29-DB35-C0EE/GraphLab-Create-License.tar.gz

import graphlab

products = graphlab.SFrame('amazon_baby.gl/')
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

print "### Use .apply() to build a new feature with the counts for each of the selected_words: ###"

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

def awesome_count(product):
    if 'awesome' in product:
        return product['awesome']
    else:
        return 0

products['awesome'] = products['word_count'].apply(awesome_count)

print "Awesome appears: ", sum(products['awesome'])

def word_count(product, word):
    if word in product:
        return product[word]
    else:
        return 0

def count_selected(selected_words, products):
    for word in selected_words:
        products[word] = products['word_count'].apply(lambda product: word_count(product, word))
        sumup = sum(products[word])
        print "Count", word, "as: ", sumup

count_selected(selected_words, products)

print "### Create a new sentiment analysis model using only the selected_words as features: ###"

products = products[products['rating'] != 3]
products['sentiment'] = products['rating'] >=4
train_data,test_data = products.random_split(.8, seed=0)
selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features= selected_words,
                                                     validation_set=test_data)

selected_words_model['coefficients']
print "Find the most positive and negative coefficients:"
selected_words_model['coefficients'].sort(['value'],ascending = False)

print "### Comparing the accuracy of different sentiment analysis model ###"

selected_words_model.evaluate(test_data)

print "### Interpreting the difference in performance between the model ###"
diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
diaper_champ_reviews['predicted_sentiment'] = selected_words_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)

print "# Most positive reviews for the Baby Trend Diaper Champ"
diaper_champ_reviews[0]['review']

diaper_champ_reviews[1]['review']

diaper_champ_reviews[-2]['review']

