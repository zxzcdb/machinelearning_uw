#  source ~/test27/bin/activate

import graphlab
from regression import simple_linear_regression
from regression import lr_get_residual_sum_of_squares
from regression import get_regression_predictions

sales = graphlab.SFrame('kc_house_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)

prices = sales['price']

test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)

(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)

print "Test Intercept: " + str(test_intercept)
print "Test Slope: " + str(test_slope)

sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Train Intercept: " + str(sqft_intercept)
print "Train Slope: " + str(sqft_slope)

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)

print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

print "Sould be 0: ", lr_get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope)

rss_prices_on_sqft = lr_get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)
