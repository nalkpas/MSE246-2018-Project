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

n_hidden_layers,n_in,n_out = 3,26,2

def get_hidden_units(n_hidden_layers,n_in,n_out):
	n_hidden_units = []
	
	if(n_hidden_layers==1):
		n_hidden_units.insert(0,(n_in*n_out)**(1/2))
		return n_hidden_units
	elif(n_hidden_layers>1):
		r = (n_in/n_out)**(1./(n_hidden_layers+1))
		print("This is r "+str(r))
		for i in range(n_hidden_layers):
			print("This is i "+str(i))
			temp_hidden_units = round(n_out*(r**(n_hidden_layers-i)))
			print("This is hidden units "+ str(temp_hidden_units))
			n_hidden_units.insert(i, temp_hidden_units)
		return n_hidden_units


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.lin1 = nn.Linear(26, 10)
		self.lin2 = nn.Linear(10, 20)
		self.lin3 = nn.Linear(20, 40)
		self.lin4 = nn.Linear(40, 20)
		self.lin5 = nn.Linear(20, 2)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.relu(self.lin4(x))
		x = F.relu(self.lin5(x))
		return x.squeeze(1)

model = Net()
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