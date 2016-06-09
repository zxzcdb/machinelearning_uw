import graphlab

people = graphlab.SFrame('people_wiki.gl/')

print "### Compare top words according to word counts to TF-IDF ###"
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
ej = people[people['name'] == 'Elton John']
ejwc = ej[['word_count']].stack('word_count', new_column_name = ['word','count'])

print "Q: What are the 3 words in his articles with highest word counts?"
print ejwc.sort('count', ascending = False)

print "Q: What are the 3 words in his articles with highest TF-IDF?"
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf
ej = people[people['name'] == 'Elton John']
print ej[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

print "### Measuring distance ###"

print "Q: What's the cosine distance between the articles on 'Elton John' and 'Victoria Beckham'?"
vb  = people[people['name'] == 'Victoria Beckham']
print graphlab.distances.cosine(ej['tfidf'][0],vb['tfidf'][0])

print "Q: What's the cosine distance between the articles on 'Elton John' and 'Paul McCartney'?"
pm = people[people['name'] == 'Paul McCartney']
print graphlab.distances.cosine(ej['tfidf'][0],pm['tfidf'][0])

print "Q: Which one of the two is closest to Elton John?"

print "### Building nearest neighbors models with different input features and setting the distance ###"

modelw = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')
modelt = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')

print "Q: What's the most similar article, other than itself, to the one on 'Elton John' using word count features?"

print modelw.query(ej)

print "Q: What's the most similar article, other than itself, to the one on 'Elton John' using TF-IDF features?"

print modelt.query(ej)

print "Q: What's the most similar article, other than itself, to the one on 'Victoria Beckham' using word count features?"

print modelw.query(vb)

print "Q: What's the most similar article, other than itself, to the one on 'Victoria Beckham' using TF-IDF features?"

print modelt.query(vb)
