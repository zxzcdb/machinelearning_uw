import graphlab

song_data = graphlab.SFrame('song_data.gl/')

print "### Counting unique users ###"

kw = song_data[song_data['artist'] == 'Kanye West']
ff = song_data[song_data['artist'] == 'Foo Fighters']
ts = song_data[song_data['artist'] == 'Taylor Swift']
lg = song_data[song_data['artist'] == 'Lady GaGa']

print "Kanye West"
print len(kw['user_id'].unique())

print "Foo Fighters"
print len(ff['user_id'].unique())

print "Taylor Swift"
print len(ts['user_id'].unique())

print "Lady GaGa"
print len(lg['user_id'].unique())

print "### Using groupby-aggregate to find the most popular and least popular artist ###"
group = song_data.groupby(key_columns='artist', operations={'total_count': graphlab.aggregate.SUM('listen_count')})

print group.sort('total_count', ascending = False)

print "### Using groupby-aggregate to find the most recommended songs ###"
train_data,test_data = song_data.random_split(.8,seed=0)
personalized_model = graphlab.item_similarity_recommender.create(train_data,
                                                         user_id='user_id',
                                                         item_id='song')
subset_test_users = test_data['user_id'].unique()[0:10000]
recommend = personalized_model.recommend(subset_test_users,k=1)
recommend.groupby(operations={'count': graphlab.aggregate.COUNT()}, key_columns='song').sort('count',ascending = False)




