import numpy as np
import pandas as pd 
import random
import pdb

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
train_data = pd.read_csv("data/nn_train.csv", index_col=0).astype(float)
train_data = train_data.drop(columns='Default?')
train_data = train_data.fillna(0)

class DefaultDataset(Dataset):
	def __init__(self, dataframe):
		self.frame = dataframe

	def __len__(self):
		return len(self.frame)

	def __getitem__(self,idx):
		covariates = self.frame.drop(columns=["GrossChargeOffAmount"])
		covariates = FloatTensor(covariates[idx:idx+1].values)
		defaults = FloatTensor(self.frame[idx:idx+1]["GrossChargeOffAmount"].values)
		sample = {"X": covariates, "Y": defaults}
		return sample

train_set = DefaultDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=False, num_workers = 2)

# define model
class Net(nn.Module):
	def __init__(self, modules):
		super(Net, self).__init__()
		for layer,module in enumerate(modules):
			self.add_module("layer_" + str(layer), module)

	def forward(self, x):
		for layer in self.children():
			x = layer(x)
		return x.squeeze(1)

# hyperparameters
n_out = 1
n_in = train_data.shape[1] - n_out
n_layers = np.random.randint(4,10)
decay = 0.02

# initialization helpers
def generate_layers(n_in,n_out):
	layers_size = {-1:n_in}
	for i in range(n_layers - 1):
		if layers_size[i -1] <= 1: break
		layers_size[i] = np.random.randint(n_out,layers_size[i - 1])
	layers_size[len(layers_size) - 1] = n_out
	return layers_size

def generate_modules(layers_size):
	modules = []
	for i in layers_size.keys():
		if i == -1: continue
		modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
		# modules.append(nn.BatchNorm1d(layers_size[i]))
	return modules

# num_trials = 200
# while counter < num_trials:

# initialize model
layers_size = generate_layers(n_in,n_out)
print(layers_size)
modules = generate_modules(layers_size)
model = Net(modules)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(),weight_decay=decay)

# training
num_epochs = 3
print_interval = 100
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]), Variable(batch["Y"]).squeeze()
		
		predictions = model(X)
		loss = criterion(predictions, Y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
		if i % print_interval == print_interval - 1:
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / print_interval))
			running_loss = 0.

# saving and loading parameters 
# model.save_state_dict("output.pt")
# model.load_state_dict(torch.load("output.pt"))

# saving the whole model
# model.save("output.pt")
# model = torch.load("model.pt")

print("done")