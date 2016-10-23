
import graphlab

loans = graphlab.SFrame('lending-club-data.gl/')

loans.column_names()

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
            'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

train_data, validation_data = loans_data.random_split(.8, seed=1)

"=== Gradient boosted tree classifier ==="

model_5 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 5)

# Select all positive and negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

# Select 2 examples from the validation set for positive & negative loans
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

# Append the 4 examples into a single dataset
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

predict = model_5.predict(sample_validation_data)

predict_prob = model_5.predict(sample_validation_data, output_type='probability')

results = model_5.evaluate(validation_data)

results['confusion_matrix']

print "=== Comparison with decision trees ==="

false_negative = 1618
false_positive = 1463

total_cost = false_negative * 10000 + false_positive * 20000
print "Total cost is: ", total_cost

print "=== Most positive & negative loans ==="

validation_data['predict_safe_loans'] = model_5.predict(validation_data)
validation_data['predictions'] = model_5.predict(validation_data, output_type="probability")

top_5 = validation_data[validation_data['predict_safe_loans'] == 1].sort('predictions', ascending=False).head(5)
print "Grades of top five loans", top_5['grades']

print "=== Effect of adding more trees ==="

model_10 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 10, verbose=False)
model_50 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 50, verbose=False)
model_100 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 100, verbose=False)
model_200 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 200, verbose=False)
model_500 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
        target = target, features = features, max_iterations = 500, verbose=False)

print "Accuracy of model 10:", model_10.evaluate(validation_data)['accuracy']
print "Accuracy of model 50:", model_50.evaluate(validation_data)['accuracy']
print "Accuracy of model 100:", model_100.evaluate(validation_data)['accuracy']
print "Accuracy of model 200:", model_200.evaluate(validation_data)['accuracy']
print "Accuracy of model 500:", model_500.evaluate(validation_data)['accuracy']

print "=== Plot the training and validation error vs. number of trees ==="

