#  source ~/test27/bin/activate

from regression import polynomial_sframe
import graphlab as gl
import matplotlib.pyplot as plt

sales = gl.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living', 'price'])

print "********Small Penalty**********"

L2_small_penalty = 1e-5

poldata = polynomial_sframe(sales['sqft_living'], 15)

poldata['price'] = sales['price']

model_0 = gl.linear_regression.create(poldata, target='price',
                                        l2_penalty= L2_small_penalty, validation_set=None)
print "Q1: What is the value of coefficient of Power_1?"

print model_0.get("coefficients")
# power_1: 103.09

print "power_1 should be 103.09."

(semi_split1, semi_split2) = sales.random_split(.5, seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

poldata1 = polynomial_sframe(set_1['sqft_living'], 15)
poldata2 = polynomial_sframe(set_2['sqft_living'], 15)
poldata3 = polynomial_sframe(set_3['sqft_living'], 15)
poldata4 = polynomial_sframe(set_4['sqft_living'], 15)

poldata1['price'] = set_1['price']
poldata2['price'] = set_2['price']
poldata3['price'] = set_3['price']
poldata4['price'] = set_4['price']

model1 = gl.linear_regression.create(poldata1, target='price',
                                        l2_penalty= L2_small_penalty, validation_set=None)
model2 = gl.linear_regression.create(poldata2, target='price',
                                        l2_penalty= L2_small_penalty, validation_set=None)
model3 = gl.linear_regression.create(poldata3, target='price',
                                        l2_penalty= L2_small_penalty, validation_set=None)
model4 = gl.linear_regression.create(poldata4, target='price',
                                        l2_penalty= L2_small_penalty, validation_set=None)

print "Coefficients of model_1 are: "
print model1.get("coefficients")
print "Coefficients of model_2 are: "
print model2.get("coefficients")
print "Coefficients of model_3 are: "
print model3.get("coefficients")
print "Coefficients of model_4 are: "
print model4.get("coefficients")

plt.plot(poldata1['power_1'], poldata1['price'], '.', poldata1['power_1'], model1.predict(poldata1), '-')

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = gl.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

# Large penalty
print "********Large Penalty**********"
l2 = 1e5

model1 = gl.linear_regression.create(poldata1, target='price',
                                        l2_penalty= l2, validation_set=None)
model2 = gl.linear_regression.create(poldata2, target='price',
                                        l2_penalty= l2, validation_set=None)
model3 = gl.linear_regression.create(poldata3, target='price',
                                        l2_penalty= l2, validation_set=None)
model4 = gl.linear_regression.create(poldata4, target='price',
                                        l2_penalty= l2, validation_set=None)

print "Coefficients of model_1 are: "
print model1.get("coefficients")
print "Coefficients of model_2 are: "
print model2.get("coefficients")
print "Coefficients of model_3 are: "
print model3.get("coefficients")
print "Coefficients of model_4 are: "
print model4.get("coefficients")
