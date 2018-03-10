import numpy as np
import pandas as pd 
import pdb

data = pd.read_csv("data/output_with_missing_v2.csv", index_col=0)
data = data.drop(columns=["BorrName", "BorrStreet", "GrossApproval", "TermInMonths", 
				 "ChargeOffDate", "CDC_Name", "CDC_Street", "LoanStatus", "ThirdPartyLender_Name", "ProjectCounty","NaicsDescription"])

numerics = data.select_dtypes(include=[np.number,'bool'])
cats = data.drop(columns=numerics.columns)
cats = cats.drop(columns=['ApprovalDate'])

lookups = {}
for column in cats.columns:
	cats[column],uniques = pd.factorize(cats[column])
	lookups[column] = dict(enumerate(uniques))

def date_hash(date):
	y,m,d = map(int,date.strip().split('-'))
	return y*10**4 + m*10**2 + d

dates = data['ApprovalDate'].copy()
for index in range(len(data)):
	dates[index:index+1] = date_hash(dates[index:index+1].values[0])

data.update(cats)
data.update(dates)

pdb.set_trace()

with open('nn_lookups.txt','w') as file:
	for column, table in lookups.items():
		file.write(str(column) + "\n")
		for token, name in table.items():
			file.write(str(token) + "," + str(name) + "\n")
		file.write("\n")

shuffled = data.sample(frac=1)
train_end_index = int(round(data.shape[0] * .8))
dev_end_index = train_end_index + int(round(data.shape[0] * .1))

train_data = data[:train_end_index]
dev_data = data[train_end_index:dev_end_index]
test_data = data[dev_end_index:]

train_data.to_csv('data/nn_train.csv',index=False)
dev_data.to_csv('data/nn_dev.csv',index=False)
test_data.to_csv('data/nn_test.csv',index=False)

