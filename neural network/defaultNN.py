import numpy as np
import pandas as pd 
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

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
		defaults = defaults[idx:idx+1]["Default?"].values
		if np.max(defaults) != 1:
			import pdb
			pdb.set_trace()
		defaults = LongTensor(defaults)
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
print("done")