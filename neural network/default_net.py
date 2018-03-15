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
n_layers = 24
n_out = 2.
n_in = x_train.shape[1]
decay = 0.03
batch_size = 100

# load data
train_set = DefaultDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 2)

# define model
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

# initialize model
model = Net(modules)
try: 
	model.load_state_dict(torch.load("default_net_" + str(n_layers) + ".pt"))
	print("loaded saved model")
except:
	print("no saved model")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay = decay)

# training
num_epochs = 25
print_interval = 100
iterations = [0]
losses = []
start = time.time()
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]), Variable(batch["Y"],requires_grad=False).squeeze()

		predictions = model(X)
		loss = criterion(predictions, Y)
		if i == 0: 
			losses.append(loss)

		# pdb.set_trace()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
		if i % print_interval == print_interval - 1:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_interval))
			iterations.append(iterations[-1] + print_interval*batch_size)
			losses.append(running_loss / print_interval)
			running_loss = 0.
end = time.time()

torch.save(model.state_dict(), "default_net_" + str(n_layers) + ".pt")
with open("default_net_" + str(n_layers) + ".csv", "w") as file:
	file.write("iteration,loss\n")
	for iteration, loss in zip(iterations, losses):
		file.write(str(iteration) + "," + str(loss) + "\n")

print("done: " + str(end - start) + "s")