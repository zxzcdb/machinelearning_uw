# source ~/test27/bin/activate

import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)

prices = sales['price']

def simple_linear_regression(input, output):
    num = input.size()

    in_sum = input.sum()
    out_sum = output.sum()

    pro = input * output
    pro_sum = pro.sum()

    in_sq = input * input
    in_sq_sum = in_sq.sum()

    avi = input.mean()
    avo = output.mean()

    # use the formula for the slope
    slope = (pro_sum - num*avi*avo)/(in_sq_sum-num*avi*avi)

    # use the formula for the intercept
    intercept = avo - avi*slope

    return (intercept, slope)

test_feature = graphlab.SArray(range(5))
test_output = graphlab.SArray(1 + 1*test_feature)

(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)

print "Test Intercept: " + str(test_intercept)
print "Test Slope: " + str(test_slope)

sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])

print "Train Intercept: " + str(sqft_intercept)
print "Train Slope: " + str(sqft_slope)

def get_regression_predictions(input, intercept, slope):
    predicted_values = input*slope + intercept
    return predicted_values

my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)
print "The estimated price for a house with %d squarefeet is $%.2f" % (my_house_sqft, estimated_price)

def get_residual_sum_of_squares(input, output, intercept, slope):
    predict = input*slope + intercept

    residuals = predict - output
    rss = residuals*residuals
    RSS = rss.sum()

    return(RSS)

print "Sould be 0: ", get_residual_sum_of_squares(test_feature, test_output, test_intercept, test_slope)

rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)
print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)

def inverse_regression_predictions(output, intercept, slope):
    estimated_feature = (output - intercept)/slope
    return estimated_feature

my_house_price = 800000
estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)
print "The estimated squarefeet for a house worth $%.2f is %d" % (my_house_price, estimated_squarefeet)

(pred_intercept, pred_slope) =  simple_linear_regression(test_data['bedrooms'], test_data['price'])
print "Intercept: " + str(pred_intercept)
print "Slope: " + str(pred_slope)

rss_prices_on_bdr = get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], pred_intercept, pred_slope)
print 'The RSS of predicting Prices based on Bedrooms is : ' + str(rss_prices_on_bdr)
