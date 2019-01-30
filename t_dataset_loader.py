import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

class DiabetesDataset(Dataset):

    def __init__(self):
        xy=np.loadtxt('data/diabetes.csv.gz',delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,0:-1])
        self.y_data=torch.from_numpy(xy[:,-1])
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


dataset=DiabetesDataset()
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
def run():
    torch.multiprocessing.freeze_support()
    print('loop')


for epoch in range(10):
    for i, data in enumerate(train_loader,0):
        inputs,labels=data

        inputs,labels=Variable(inputs),Variable(labels)
        print(epoch,inputs.data,labels.data)
        if __name__ == '__main__':
#            freeze_support()
            pass
