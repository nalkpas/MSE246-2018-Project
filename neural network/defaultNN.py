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

n_layers,n_in,n_out = 5,26,2

def get_hidden_units(n_layers,n_in,n_out):
	n_hidden_units = []
	n_hidden_layers = n_layers-2
	n_hidden_units.insert(0,n_in)
	if(n_hidden_layers ==0):
		n_hidden_units.insert(n_layers-1,n_out)
		return n_hidden_units
	if(n_hidden_layers==1):
		n_hidden_units.insert(1,int((n_in*n_out)**(.1/2)))
		n_hidden_units.insert(n_layers-1,n_out)
		return n_hidden_units
	elif(n_hidden_layers>1):
		r = (n_in/n_out)**(1./(n_hidden_layers+1))
		for i in range(n_hidden_layers):
			temp_hidden_units = int(n_out*(r**(n_hidden_layers-i)))
			n_hidden_units.insert(i+1, temp_hidden_units)
		n_hidden_units.insert(n_layers-1,n_out)
		return n_hidden_units

class Net(nn.Module):
	def __init__(self, n_layers, n_in, n_out):
		super(Net, self).__init__()
		n_hidden_units = get_hidden_units(n_layers, n_in, n_out)
		for i in range(n_layers-1):
			temp_nn = nn.Linear(n_hidden_units[i],n_hidden_units[i+1])
			self.add_module("linear"+str(i),temp_nn)

	def forward(self, x):
		nn_list= self.children()
		for nn in nn_list:
			x = F.relu(nn(x))
		return x.squeeze(1)

model = Net(n_layers,n_in,n_out)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_data = pd.read_csv("train.csv")
train_data = train_data.drop(columns=["BorrName", "BorrStreet", "GrossApproval", "TermInMonths", 
				 "ChargeOffDate", "GrossChargeOffAmount_Norm", "GrossChargeOffAmount_Norm", 
				 "LoanStatus"])

class DefaultDataset(Dataset):
	def __init__(self, dataframe):
		self.frame = dataframe

	def __len__(self):
		return len(self.frame)

	def __getitem__(self,idx):
		covariates = self.frame.drop(columns=["Default?"])
		covariates = FloatTensor(covariates[idx:idx+1].values)
		defaults = LongTensor(self.frame[idx:idx+1]["Default?"].values*1)
		sample = {"X": covariates, "Y": defaults}
		return sample

train_set = DefaultDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers = 2)

num_epochs = 2
print_interval = 100
for epoch in range(num_epochs):
	running_loss = 0.
	for i, batch in enumerate(train_loader):
		X, Y = Variable(batch["X"]), Variable(batch["Y"]).squeeze()
		
		if np.max(Y.data.numpy()) > 1:
			pdb.set_trace()
		
		optimizer.zero_grad()
		predictions = model(X)
		loss = criterion(predictions, Y)
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