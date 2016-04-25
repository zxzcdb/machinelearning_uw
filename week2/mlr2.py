import graphlab as gl
import graphlab.numpy
import numpy as np

sales = gl.SFrame('kc_house_data.gl/')

def get_numpy_data(data, features, output):

    data['constant'] = 1

    features.append('constant')
    ft = data.select_columns(features)
    ftmatrix = ft.to_numpy()

#    out = gl.SArray(data[output])
#    outarray = gl.to_numpy(out)

    outarray = np.asarray(data[output])

    return(ftmatrix, outarray)

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')

print example_features[0,:]
print example_output[0]
print example_output

my_weights = np.array([1., 1.]) # the example weights
my_features = example_features[0,] # we'll use the first data point
predicted_value = np.dot(my_features, my_weights)
print predicted_value

def predict_output(fm, wei):
    predictions = np.dot(fm, wei)
    return(predictions)

test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0
