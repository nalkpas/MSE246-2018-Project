import numpy as np
import pandas as pd 
import random
import pdb
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# load data 
# train_data = pd.read_csv("data/nn_train.csv", index_col=0)
# train_data = train_data.drop(columns='GrossChargeOffAmount')
# train_data = train_data.fillna(0)

# categorical_vars = ['CDC_State', 'ThirdPartyLender_State', 'DeliveryMethod', 'subpgmdesc', 'ProjectState', 'BusinessType', 
# 					'TermMultipleYear', 'RepeatBorrower', 'BankStateneqBorrowerState', 'ProjectStateneqBorrowerState', '2DigitNaics']
# for var in categorical_vars:
# 	train_data[var] = train_data[var].astype('category')
# train_data = pd.get_dummies(train_data, columns = train_data.columns.delete(train_data.columns.get_loc('LoanStatus')))

# load data

train_data = pd.read_csv("data/random_train0310.csv")
test_data = pd.read_csv("data/random_test0310.csv")
val_data = pd.read_csv("data/random_val0310.csv")

train_data = train_data[train_data.LoanStatus != "MISSING"]
test_data = test_data[test_data.LoanStatus != "MISSING"]
val_data = val_data[val_data.LoanStatus != "MISSING"]

c = train_data.columns.tolist()
dropped_columns = ['LoanStatus', 'ChargeOffDate', 'GrossChargeOffAmount', 'BorrZip', 'CDC_Zip', 'BorrCity',
                   'CDC_City', 'ThirdPartyLender_City', 'ProjectCounty', 'ApprovalDate']
for col in dropped_columns:
    c.remove(col)

x_train = train_data[c]
x_test = test_data[c]
x_val = val_data[c]

print(train_data.shape)

numerics = train_data.select_dtypes(include=[np.number,'bool'])
cats = train_data.drop(columns=numerics.columns)
x_train = pd.get_dummies(train_data, columns = cats.columns)
test_data = pd.get_dummies(test_data, columns = cats.columns)
val_data = pd.get_dummies(val_data, columns = cats.columns)

print(train_data.shape)
for column in train_data.columns:
	print(column)
pdb.set_trace()

class DefaultDataset(Dataset):
	def __init__(self, dataframe):
		self.frame = dataframe

	def __len__(self):
		return len(self.frame)

	def __getitem__(self,idx):
		covariates = self.frame.drop(columns=['LoanStatus'])
		covariates = FloatTensor(covariates[idx:idx+1].values)
		defaults = LongTensor((self.frame.LoanStatus[idx:idx+1].values == 'CHGOFF')*1).squeeze()
		sample = {"X": covariates, "Y": defaults}
		return sample

train_set = DefaultDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers = 2)

# hyperparameters
n_layers = 3
n_out = 2.
n_in = train_data.shape[1] - 1
decay = 1.

# define model
layers_size = {-1: n_in}
factor = (n_out/n_in)**(1/n_layers)
for layer in range(n_layers):
	layers_size[layer] = int(np.rint(n_in * factor**(layer + 1)))
modules = []
print(layers_size)

for i in layers_size.keys():
	if i == -1: continue
	modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
	# if i == 0:
	# 	modules.append(nn.BatchNorm1d(layers_size[i]))
	modules.append(nn.ReLU())

class Net(nn.Module):
	def __init__(self, modules):
		super(Net, self).__init__()
		for layer,module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for i, layer in enumerate(self.children()):
			x = layer(x)
		return x.squeeze()

# initialize model
model = Net(modules)
try: 
	model.load_state_dict(torch.load("default_net_" + str(n_layers) + ".pt"))
	print("loaded saved model")
except:
	print("no saved model")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# training
num_epochs = 2
print_interval = 100
start = time.time()
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]), Variable(batch["Y"],requires_grad=False)

		predictions = model(X)
		loss = criterion(predictions, Y)

		# pdb.set_trace()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
		if i % print_interval == print_interval - 1:
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / print_interval))
			running_loss = 0.
end = time.time()

torch.save(model.state_dict(), "default_net_" + str(n_layers) + ".pt")

print("done: " + str(end - start) + "s")