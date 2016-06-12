import graphlab

image_train = graphlab.SFrame('image_train_data/')
image_test = graphlab.SFrame('image_test_data/')

print "### Computing summary statistics of the data ###"

image_train['label'].sketch_summary()

print "### Creating category-specific image retrieval models ###"
automobile = image_train[image_train['label' == 'automobile']]
cat = image_train[image_train['label' == 'cat']]
dog = image_train[image_train['label' == 'dog']]
bird = image_train[image_train['label' == 'bird']]

auto_model = graphlab.graphlab.nearest_neighbors.create(automobile,
                                                         features=['deep_features'],
                                                         label='id')
cat_model = graphlab.graphlab.nearest_neighbors.create(cat,
                                                         features=['deep_features'],
                                                         label='id')
dog_model = graphlab.graphlab.nearest_neighbors.create(dog,
                                                         features=['deep_features'],
                                                         label='id')
bird_model = graphlab.graphlab.nearest_neighbors.create(bird,
                                                         features=['deep_features'],
                                                         label='id')

print "Q: Nearest cat."
cat_id = cat_model.query(image_test[0:1])['reference_label'][0]
image_train[cat_id]['image'].show()
print "Q: Nearest dog."
dog_model.query(image_test[0:1])

print "### A simple example of nearest-neighbors classification ###"
print "cat mean"
cat_model.query(image_test[0:1])['distance'][1:5].mean()
# 36.53873358521806
print "dog mean"
dog_model.query(image_test[0:1])['distance'][1:5].mean()
# 37.84732348269601

print "### [Challenging Question] Computing nearest neighbors accuracy using SFrame operations ###"
image_test_cat = image_test[image_test['label' == 'cat']]
image_test_dog = image_test[image_test['label' == 'dog']]
image_test_bird = image_test[image_test['label' == 'bird']]
image_test_automobile = image_test[image_test['label' == 'automobile']]


dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
dog_car_neighbors = auto_model.query(image_test_dog, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)
dog_dog_neighbors = dog_model.query(image_test_dog, k=1)


dog_distances = graphlab.SFrame({'dog-automobile': dog_car_neighbors['distance'],'dog-bird': dog_bird_neighbors['distance'], 'dog-cat': dog_cat_neighbors['distance'], 'dog-dog': dog_dog_neighbors['distance']})

dog_distances.head()

def is_dog_correct(row):
    if row['dog-dog'] < row['dog-automobile'] and row['dog-dog'] < row['dog-bird'] and row['dog-dog'] < row['dog-cat']:
        return 1
    else:
        return 0

print "Is dog correct:", is_dog_correct(dog_distances[0:1])

print "Correct ratio:", dog_distances.apply(is_dog_correct).sum()/dog_distances.num_rows()
