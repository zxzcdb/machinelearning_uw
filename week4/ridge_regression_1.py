#  source ~/test27/bin/activate

from regression import polynomial_sframe
from regression import poly_regression
from regression import model_co
import graphlab as gl
import matplotlib as mpl
import numpy as np

def polynomial_sframe(feature, degree):
    if degree < 1:
        print "ERROR: degree < 1."
    poly_sframe = gl.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name] = feature.apply(lambda x: x**power)
    return poly_sframe

def ridge_regression(data, l2):
    model = gl.linear_regression.create(data, target = 'price',
                                           l2_penalty = l2, validation_set = None)
    return(model)

def model_co(model):
    print(model.get("coefficients"))

sales = gl.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living','price'])

def get_residual_sum_of_squares(model, data, outcome):
    prediction = model.predict(data)
    residuals = prediction - outcome
    RSS = residuals*residuals
    RSS = RSS.sum()
    return(RSS)

L2_small_penalty = 1e-5

poldata = polynomial_sframe(sales['sqft_living'],15)

poldata['price'] = sales['price']

model0 = poly_regression(poldata,L2_small_penalty)

model0.get("coefficients")
# power_1: 103.09

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

poldata1 = polynomial_sframe(set_1['sqft_living'],15)
poldata2 = polynomial_sframe(set_2['sqft_living'],15)
poldata3 = polynomial_sframe(set_3['sqft_living'],15)
poldata4 = polynomial_sframe(set_4['sqft_living'],15)

poldata1['price'] = set_1['price']
poldata2['price'] = set_2['price']
poldata3['price'] = set_3['price']
poldata4['price'] = set_4['price']

model1 = poly_regression(poldata1,L2_small_penalty)
model2 = poly_regression(poldata2,L2_small_penalty)
model3 = poly_regression(poldata3,L2_small_penalty)
model4 = poly_regression(poldata4,L2_small_penalty)

model_co(model1)
model_co(model2)
model_co(model3)
model_co(model4)
#plt.plot(poldata1['power_1'],poldata1['price'],'.',poldata1['power_1'], model1.predict(poldata1),'-')

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = gl.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

def slice_data(data,i,k):
    if i <= 0:
        print "ERROR: i<=0!"
    if i > k:
        print "ERROR: i>k!"

    n = len(data)
    start = (n*i)/k
    end = (n*(i+1))/k-1
    return(data[start:end+1])

validation4 = slice_data(train_valid_shuffled,4,10)

print int(round(validation4['price'].mean(), 0))

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    feature = data[features_list]
    poldata = polynomial_sframe(feature[0],15)
    poldata[output_name] = data[output_name]
    rss = 0
    for i in range(0,k-1):
        start = (n*i)/k
        end = (n*(i+1))/k
        vali = data[start:end+1]
        train = data[0:start] + data[end+1:len(data)]
        model = gl.linear_regression.create(train, target = 'price',
                                           l2_penalty = l2_penalty, validation_set = None)
        rss = get_residual_sum_of_squares(model, vali, 'price')
        rss+=rss
    return rss

for l2_penalty in np.logspace(1, 7, num=13):
    print(l2_penalty)
    rss = k_fold_cross_validation(10, l2_penalty, train_valid_shuffled, 'price', ['sqft_living'])
    print rss
