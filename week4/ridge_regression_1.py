#  source ~/test27/bin/activate

from regression import polynomial_sframe
from regression import ridge_regression
from regression import ridge_regression_gradient_descent
from regression import model_co
import graphlab as gl
import matplotlib as mpl
import numpy as np

sales = gl.SFrame('kc_house_data.gl/')

sales = sales.sort(['sqft_living','price'])

L2_small_penalty = 1e-5

poldata = polynomial_sframe(sales['sqft_living'],15)

poldata['price'] = sales['price']

model0 = ridge_regression(poldata,L2_small_penalty)

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

model1 = ridge_regression(poldata1,L2_small_penalty)
model2 = ridge_regression(poldata2,L2_small_penalty)
model3 = ridge_regression(poldata3,L2_small_penalty)
model4 = ridge_regression(poldata4,L2_small_penalty)

model_co(model1)
model_co(model2)
model_co(model3)
model_co(model4)
#plt.plot(poldata1['power_1'],poldata1['price'],'.',poldata1['power_1'], model1.predict(poldata1),'-')

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = gl.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

# Large penalty
l2 = 1e5

model1 = ridge_regression(poldata1,l2)
model2 = ridge_regression(poldata2,l2)
model3 = ridge_regression(poldata3,l2)
model4 = ridge_regression(poldata4,l2)

model_co(model1)
model_co(model2)
model_co(model3)
model_co(model4)
