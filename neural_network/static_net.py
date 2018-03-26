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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers = 2)

# hyperparameters
n_out = 1
n_in = train_data.shape[1] - n_out
n_layers = np.random.randint(4,10)
decay = 1

# define model
layers_size = {-1: n_in, 0: 2*n_in, 1: int(np.ceil(1.5*n_in)), 2: n_in, 3: n_in, 4: int(np.ceil(0.75*n_in)), 
				5: int(np.ceil(0.5*n_in)), 6: int(np.ceil(0.25*n_in)), 7: 3, 7: n_out}
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.linear1 = nn.Linear(layers_size[-1],layers_size[0])
		self.linear2 = nn.Linear(layers_size[0],layers_size[1])
		self.linear3 = nn.Linear(layers_size[1],layers_size[2])
		self.linear4 = nn.Linear(layers_size[2],layers_size[3])
		self.linear5 = nn.Linear(layers_size[3],layers_size[4])
		self.linear6 = nn.Linear(layers_size[4],layers_size[5])
		self.linear7 = nn.Linear(layers_size[5],layers_size[6])
		self.linear8 = nn.Linear(layers_size[6],layers_size[7])

		# self.batch_norm1 = nn.BatchNorm1d(num_features=layers_size[0])
		# self.batch_norm2 = nn.BatchNorm1d(num_features=layers_size[1])
		# self.batch_norm3 = nn.BatchNorm1d(num_features=layers_size[2])

	def forward(self, x):
		# x = F.relu(self.batch_norm1(self.linear1(x)))
		# # x = F.relu(self.batch_norm2(self.linear2(x)))
		# # x = F.relu(self.batch_norm3(self.linear3(x)))
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		x = F.relu(self.linear4(x))
		x = F.relu(self.linear5(x))
		x = F.relu(self.linear6(x))
		x = F.relu(self.linear7(x))
		x = F.relu(self.linear8(x))
		return x

# initialize model
model = Net()
try: 
	model.load_state_dict(torch.load("static_net.pt"))
	print("loaded saved model")
except:
	print("no saved model")
criterion = nn.SmoothL1Loss()
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

torch.save(model.state_dict(), "static_net.pt")

print("done: " + str(end - start) + "s")