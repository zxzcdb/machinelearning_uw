# source ~/test27/bin/activate
#%matplotlib inline

import graphlab

import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    if degree < 1:
        print "ERROR: degree < 1."
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
    # then loop over the remaining degrees:
    # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe

sales = graphlab.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

# Model 1
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)

#let's take a look at the weights before we plot
model1.get("coefficients")

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')

# Model 2
poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)

model2.get("coefficients")

plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')

# Model 3
poly3_data = polynomial_sframe(sales['sqft_living'], 3)

my_features3 = poly3_data.column_names() # get the name of the features
print my_features3
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features3, validation_set = None)

model3.get("coefficients")

plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')

# Changing the data and re-learning¶

(sub_1, sub_2) = sales.random_split(0.5, seed=0)
sub_1_1, sub_1_2 = sub_1.random_split(0.5, seed=0)
sub_2_1, sub_2_2 = sub_2.random_split(0.5, seed=0)

def polynomial_fit(data, degree):
    poldata = polynomial_sframe(data['sqft_living'],degree)
    feature = poldata.column_names()

    poldata['price'] = data['price']

    model = graphlab.linear_regression.create(poldata, target = 'price', features = feature, validation_set = None)
    model.get("coefficients")

    plt.plot(poldata['power_1'],poly3_data['price'],'.',poldata['power_1'], model.predict(poldata),'-')

polynomial_fit(sub_1_1, 15)

polynomial_fit(sub_1_2, 15)

polynomial_fit(sub_2_1, 15)

polynomial_fit(sub_2_2, 15)

#Selecting a Polynomial Degree
train_validation_data, test_data =sales.random_split(0.9, seed=1)
train_data, validation_data = train_validation_data.random_split(0.5, seed=1)

from regression import get_residual_sum_of_squares as rss

for degree in range(1,15+1):
    poldata = polynomial_sframe(train_data['sqft_living'], degree)

    feature = poldata.column_names()
    print feature

    poldata['price'] = train_data['price']
    model = graphlab.linear_regression.create(poldata, target = 'price', features = feature, verbose = False, validation_set = None)

    model.get("cofficients")

    valdata = polynomial_sframe(validation_data['sqft_living'], degree)
    valdata['price'] = validation_data['price']

    print 'RSS', rss(model, valdata, valdata['price'])
