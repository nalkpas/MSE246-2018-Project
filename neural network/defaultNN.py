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

###### load data ######
train_data = pd.read_csv("nn_train.csv", index_col=0).astype(float)

class DefaultDataset(Dataset):
	def __init__(self, dataframe):
		self.frame = dataframe

	def __len__(self):
		return len(self.frame)

	def __getitem__(self,idx):
		covariates = self.frame.drop(columns=["Default?","GrossChargeOffAmount"])
		covariates = FloatTensor(covariates[idx:idx+1].values)
		defaults = LongTensor(self.frame[idx:idx+1]["Default?"].values*1)
		sample = {"X": covariates, "Y": defaults}
		return sample

train_set = DefaultDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers = 2)

class Net(nn.Module):
	def __init__(self, layers_size):
		super(Net, self).__init__()
		self.layers = []
		for layer in layers_size.items():
			temp_nn = nn.Linear(n_hidden_units[i],n_hidden_units[i+1])
			self.add_module("linear"+str(i),temp_nn)

	def forward(self, x):
		nn_list= self.children()
		for nn in nn_list:
			x = F.relu(nn(x))
		return x.squeeze(1)

pdb.set_trace()
n_out = 3
n_in = train_data.shape[1]
n_layers = np.random.randint(5,8)

layers_size = {0:np.random.randint(n_out,n_in)}

for i in range(1,n_layers):
	layers_size[i] = np.random.randint(n_out,layers_size[i - 1] + 2)

model = Net(n_layers,n_in,n_out)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 2
print_interval = 100
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]), Variable(batch["Y"]).squeeze()
		pdb.set_trace()
		
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