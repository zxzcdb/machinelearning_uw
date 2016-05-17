# source ~/test27/bin/activate
#%matplotlib inline

import graphlab
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from regression import polynomial_sframe
from regression import polynomial_fit
from regression import get_residual_sum_of_squares as rss

sales = graphlab.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

# Model 1: 1 feature & degree 1
poly1_data = polynomial_sframe(sales['sqft_living'], 1)

poly1_data['price'] = sales['price'] # add price to the data since it's the target

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)

#let's take a look at the weights before we plot
print "Coefficients of model with 1 degree:"

model1.get("coefficients")

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')

# Model 2: 1 feature & degree 2
poly2_data = polynomial_sframe(sales['sqft_living'], 2)

my_features = poly2_data.column_names() # get the name of the features

poly2_data['price'] = sales['price'] # add price to the data since it's the target

model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)

print "Coefficients of model with 2 degree:"

model2.get("coefficients")

plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')

# Model 3:  1 feature & degree 3
poly3_data = polynomial_sframe(sales['sqft_living'], 3)

my_features3 = poly3_data.column_names() # get the name of the features

print "Features of model3 (degree 3) is: ", my_features3

poly3_data['price'] = sales['price'] # add price to the data since it's the target

model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features3, validation_set = None)

print "Coefficients of model with 1 degree:"
model3.get("coefficients")

plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')

# Changing the data and relearning

(sub_1, sub_2) = sales.random_split(0.5, seed=0)
sub_1_1, sub_1_2 = sub_1.random_split(0.5, seed=0)
sub_2_1, sub_2_2 = sub_2.random_split(0.5, seed=0)

polynomial_fit(sub_1_1, 15)

polynomial_fit(sub_1_2, 15)

polynomial_fit(sub_2_1, 15)

polynomial_fit(sub_2_2, 15)

#Selecting a Polynomial Degree
train_validation_data, test_data =sales.random_split(0.9, seed=1)
train_data, validation_data = train_validation_data.random_split(0.5, seed=1)

for degree in range(1,15+1):
    poldata = polynomial_sframe(train_data['sqft_living'], degree)

    feature = poldata.column_names()
    print "Feature of model with degree[", degree, "] is: ", feature

    poldata['price'] = train_data['price']

    model = graphlab.linear_regression.create(poldata, target = 'price', features = feature, verbose = False, validation_set = None)

    print "Coefficients of model with degree[", degree, "] is: ",

    model.get("coefficients")

    valdata = polynomial_sframe(validation_data['sqft_living'], degree)

    valdata['price'] = validation_data['price']

    print 'RSS of model with degree[', degree, "]degree is: ", rss(model, valdata, valdata['price'])
