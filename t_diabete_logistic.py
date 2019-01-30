import torch
from torch.autograd import Variable
import numpy as np

xy=np.loadtxt('data/diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data=Variable(torch.from_numpy(xy[:,0:-1]))
y_data=Variable(torch.from_numpy(xy[:,-1]))
#x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
#y_data = Variable(torch.from_numpy(xy[:, [-1]]))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        out1=self.sigmoid(self.linear1(x_data))
        out2=self.sigmoid(self.linear2(out1))
        y_pred=self.sigmoid(self.linear3(out2))
        return y_pred
model=Model()

criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(1000):
    y_pred=model.forward(x_data)
    loss=criterion(y_pred,y_data)
    loss.backward()
    optimizer.zero_grad()
    optimizer.step()
    print("Epoch",epoch,'loss=',loss.data)
