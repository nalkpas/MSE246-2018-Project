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

# hyperparameters
n_layers = 7
decay = 0.1
batch_size = 20
k = 2
num_epochs = 5

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

def add_missing_columns(d1, d2):
    missing_cols_2 = (set(d1.columns) - set(d2.columns))
    missing_cols_1 = (set(d2.columns) - set(d1.columns))
    for c in missing_cols_2:
        d2[c] = 0
    for c in missing_cols_1:
        d1[c] = 0
    return d1, d2

x_train, x_test = add_missing_columns(x_train, x_test)
x_train, x_val = add_missing_columns(x_train, x_val)
x_test, x_val = add_missing_columns(x_test, x_val)

print(x_train.shape)

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

# shape data
n_out = 2.
n_in = x_train.shape[1]


# load data
train_set = DefaultDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 2)

# define model
layers_size = {-1: n_in}
factor = (n_out/k/n_in)**(1/(n_layers - 1))
for layer in range(n_layers):
	layers_size[layer] = int(np.rint(k*n_in * factor**(layer)))
print(layers_size)

modules = []
for i in layers_size.keys():
	if i == -1: continue
	modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
	if i < n_layers - 1:
		modules.append(nn.BatchNorm1d(layers_size[i]))
		modules.append(nn.ReLU())
		modules.append(nn.Dropout(0.15))

class Net(nn.Module):
	def __init__(self, modules):
		super(Net, self).__init__()
		for layer,module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for layer in self.children():
			x = layer(x)
		return x

# initialize model
model = Net(modules)
try: 
	model.load_state_dict(torch.load("models/default_net_" + str(n_layers) + "-" + str(k) + ".pt"))
	print("loaded saved model")
except:
	print("no saved model")
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=decay)
model.train()

# training
print_interval = 100
iterations = [0]
losses = []
start = time.time()
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]).squeeze(), Variable(batch["Y"],requires_grad=False).squeeze()

		predictions = model(X)
		loss = criterion(predictions, Y)
		# if epoch == 4:
		# 	fu = list(model.parameters())
		# 	pdb.set_trace()
		if i == 0: 
			losses.append(loss.data[0])

		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), max_norm=2, norm_type=2)
		optimizer.step()

		running_loss += loss.data[0]
		if i % print_interval == print_interval - 1:
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_interval))
			iterations.append(iterations[-1] + print_interval*batch_size)
			losses.append(running_loss / print_interval)
			running_loss = 0.
end = time.time()

torch.save(model.state_dict(), "models/default_net_" + str(n_layers) + "-" + str(k) + ".pt")
with open("loss_graphs/default_net_" + str(n_layers) + "-" + str(k) + ".csv", "w") as file:
	file.write("iteration,loss\n")
	for iteration, loss in zip(iterations, losses):
		file.write(str(iteration) + "," + str(loss) + "\n")

print("done: " + str(end - start) + "s")