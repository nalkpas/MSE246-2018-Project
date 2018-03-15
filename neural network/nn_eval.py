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

# process data
train_data = pd.read_csv("data/random_train0314.csv")
test_data = pd.read_csv("data/random_test0314.csv")
val_data = pd.read_csv("data/random_val0314.csv")

c = train_data.columns.tolist()
dropped_columns = ['LoanStatus', 'ChargeOffDate', 'GrossChargeOffAmount', 'BorrZip', 'CDC_Zip', 'BorrCity',
				   'CDC_City', 'ThirdPartyLender_City', 'ProjectCounty', 'ApprovalDate']
for col in dropped_columns:
	c.remove(col)

x_train = train_data[c]
x_test = test_data[c]
x_val = val_data[c]

numerics = x_train.select_dtypes(include=[np.number,'bool'])
cats = x_train.drop(columns=numerics.columns)
x_train = pd.get_dummies(x_train, columns = cats.columns)
x_test = pd.get_dummies(x_test, columns = cats.columns)
x_val = pd.get_dummies(x_val, columns = cats.columns)

print(x_train.shape)

y_train = (train_data['LoanStatus'].values == "CHGOFF")*1
y_test = (test_data['LoanStatus'].values == "CHGOFF")*1
y_val = (val_data['LoanStatus'].values == "CHGOFF")*1

print(y_train.shape)

class DefaultDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.y)

	def __getitem__(self,idx):
		covariates = FloatTensor(self.x[idx:idx+1].values)
		defaults = LongTensor(self.y[idx:idx+1])
		sample = {"X": covariates, "Y": defaults}
		return sample

# hyperparameters
n_layers = 7
n_out = 2.
n_in = x_train.shape[1]
decay = 0.03
batch_size = 100

# load data
val_set = DefaultDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(y_val), shuffle=False, num_workers = 2)

# initialize model
layers_size = {-1: n_in}
factor = (n_out/n_in)**(1/(n_layers))
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

model = Net(modules)
try: 
	model.load_state_dict(torch.load("default_net_" + str(n_layers) + ".pt"))
	print("loaded saved model")
except:
	print("no saved model")
model.eval()

for batch in val_loader:
	X, Y = Variable(batch["X"]), Variable(batch["Y"],requires_grad=False).squeeze()
	pdb.set_trace()
	val_preds = model(X)
