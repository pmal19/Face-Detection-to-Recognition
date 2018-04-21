import time
import os
import sys
import torch
from torch import np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
import pdb

# sys.path.insert(0, '../../')
sys.path.insert(0, '../SFD_pytorch')

from net_s3fd import *
from recogWithoutDetectionLayers import *


def save(model, optimizer, loss, filename):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.data[0]
        }
    torch.save(save_dict, filename)



def loadNet(numberOfClasses, modelPath):
	recogModel = recog(numberOfClasses)
	loadedModel = torch.load(modelPath)
	newModel = recogModel.state_dict()
	pretrained_dict = {k: v for k, v in loadedModel.items() if k in newModel}
	newModel.update(pretrained_dict)
	recogModel.load_state_dict(newModel)
	return recogModel


class faceDataset(Dataset):
    def __init__(self, csvPath, imagesPath, transform = None):
    
        self.data = pd.read_csv(csvPath)
        self.imagesPath = imagesPath
        self.transform = transform
        self.imagesData = self.data['Image_Name']
        self.data['Tags'] = self.data['Tags'].apply(lambda x: re.sub(r'([^0-9]|_)+', '', x))
        self.labelsData = self.data['Tags'].astype('int')

    def __getitem__(self, index):
        imageName = os.path.join(self.imagesPath,self.data.iloc[index, 0])
        image = Image.open(imageName + '.jpg')
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labelsData[index] - 1
        return image, label

    def __len__(self):
        return len(self.data)


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, use_cuda):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #pdb.set_trace()
        # data = data.view(4, 3, 32, 32)
        if(use_cuda):
        	data, target = Variable(data.cuda()), Variable(target.cuda())
        else:
        	data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0]))
            save(model, optimizer, loss, 'faceRecog.saved.model')
        print(batch_idx,loss.data[0])    	

def train(numEpochs, trainLoader, model, optimizer, criterion, use_cuda):
    for epoch in range(numEpochs):
        trainEpoch(epoch, 20000000, trainLoader, model, optimizer, criterion, use_cuda)


def main():

    batchSize = 16
    epochs = 5
    learningRate = 0.01
    momentum = 0.9

    numWorkers = 5
    numEpochs = 10

    numberOfClasses = 10

    modelPath = '../SFD_pytorch/s3fd_convert.pth'

    use_cuda = torch.cuda.is_available()
    print("Using GPU - ",use_cuda)

    net = loadNet(numberOfClasses, modelPath)
    if(use_cuda):
    	net.cuda()	
    

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr = learningRate)
    # optimizer = optim.SGD(net.parameters(), lr = learningRate, momentum = momentum)
    optimizer = optim.Adam(net.parameters(), lr = learningRate)

    imagesPath = '../Toy Dataset/Celeb_Small_Dataset/'
    trainData = '../Toy Dataset/index.csv'
    # testData = 'kaggleamazon/test.csv'

    transformations = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

    trainingDataset = faceDataset(trainData, imagesPath, transformations)
    # testingDataset = faceDataset(testData,imagesPath)


    trainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)
    # testLoader = DataLoader(testingDataset,batchSize,num_workers=numWorkers)

    # torch.backends.cudnn.benchmark = True
    optimizer = optim.Adam(net.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, net, optimizer, criterion, use_cuda)

    testImage1 = transformations(Image.open('../Toy Dataset/Test/Celeb_4/25-FaceId-0.jpg').convert('RGB'))
    testImage2 = transformations(Image.open('../Toy Dataset/Test/Celeb_4/26-FaceId-0.jpg').convert('RGB'))
    testImage3 = transformations(Image.open('../Toy Dataset/Test/Celeb_3/27-FaceId-0.jpg').convert('RGB'))

    if(use_cuda):
    	testImage1 = Variable(testImage1.cuda()).unsqueeze(0)
    	testImage2 = Variable(testImage2.cuda()).unsqueeze(0)
    	testImage3 = Variable(testImage3.cuda()).unsqueeze(0)
    else:
    	testImage1 = Variable(testImage1).unsqueeze(0)
    	testImage2 = Variable(testImage2).unsqueeze(0)
    	testImage3 = Variable(testImage3).unsqueeze(0)

    output1 = net(testImage1)
    output2 = net(testImage2)
    output3 = net(testImage2)
    print("testImage1 - ",output1)
    print("testImage2 - ",output2)
    print("testImage3 - ",output3)
    enc1 = net.getEncoding(testImage1)
    enc2 = net.getEncoding(testImage2)
    enc3 = net.getEncoding(testImage3)
    print("enc1 - ",enc1)
    print("enc2 - ",enc2)
    print("enc3 - ",enc3)
    distance12 = net.isSame(testImage1, testImage2)
    distance13 = net.isSame(testImage1, testImage3)
    distance23 = net.isSame(testImage2, testImage3)
    print("distance12 - ",distance12)
    print("distance13 - ",distance13)
    print("distance23 - ",distance23)






if __name__ == "__main__":
    main()
