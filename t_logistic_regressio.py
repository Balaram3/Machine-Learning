import torch
from torch.autograd import Variable


class Model(torch.nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
y_data=Variable(torch.Tensor([[0.0],[0.0],[1.0],[1.0]]))


model=Model()
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.005)

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, loss.data)
hour_var = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
hour_var = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)
