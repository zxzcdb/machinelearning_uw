# source ~/test27/bin/activate
# pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/1.9/klhscott@163.com/FCEF-BFF5-FBDF-57DD-620D-9B29-DB35-C0EE/GraphLab-Create-License.tar.gz

import graphlab

sales = graphlab.SFrame('home_data.gl/')

print "sales: ", sales

train_data,test_data = sales.random_split(.8,seed=0)

sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)

print test_data['price'].mean()

print sqft_model.evaluate(test_data)

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

sales[my_features].show()

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)

print my_features

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

print "=== house1 ==="

house1 = sales[sales['id']=='5309101200']
house1
print house1['price']
print sqft_model.predict(house1)
print "Predict 1", my_features_model.predict(house1)

print "=== house2 ==="

house2 = sales[sales['id']=='1925069082']
house2
print sqft_model.predict(house2)
print "Predict 2", my_features_model.predict(house2)

print "=== Last house, super fancy ==="

bill_gates = {'bedrooms':[8],
              'bathrooms':[25],
              'sqft_living':[50000],
              'sqft_lot':[225000],
              'floors':[4],
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}

print "Predict 3", my_features_model.predict(graphlab.SFrame(bill_gates))
