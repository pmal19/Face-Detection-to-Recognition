import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# class L2Norm(nn.Module):
#     def __init__(self,n_channels, scale=1.0):
#         super(L2Norm,self).__init__()
#         self.n_channels = n_channels
#         self.scale = scale
#         self.eps = 1e-10
#         self.weight = nn.Parameter(torch.Tensor(self.n_channels))
#         self.weight.data *= 0.0
#         self.weight.data += self.scale

#     def forward(self, x):
#         norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
#         x = x / norm * self.weight.view(1,-1,1,1)
#         return x

class recog(nn.Module):
    def __init__(self, numberOfClasses):
        super(recog, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6     = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7     = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.biLin1 = nn.Bilinear(256*16*16, 512*8*8, 128)
        self.biLin2 = nn.Bilinear(512*4*4, 1024*6*6, 128)
        self.biLin3 = nn.Bilinear(512*3*3, 256*2*2, 32)

        self.biLin4 = nn.Bilinear(128, 128, 32)
        self.biLin5 = nn.Bilinear(32, 32, 16)

        self.features = nn.Linear(16, 10)
        self.classifier = nn.Linear(10, numberOfClasses)

        # self.conv3_3_norm = L2Norm(256,scale=10)
        # self.conv4_3_norm = L2Norm(512,scale=8)
        # self.conv5_3_norm = L2Norm(512,scale=5)

        # self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        # self.conv3_3_norm_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        # self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        # self.conv4_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        # self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        # self.conv5_3_norm_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        # self.fc7_mbox_conf     = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        # self.fc7_mbox_loc      = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        # self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        # self.conv6_2_mbox_loc  = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        # self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        # self.conv7_2_mbox_loc  = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h)); f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h)); f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h)); f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h));     ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h)); f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h)); f7_2 = h

        print(f3_3.size())
        print(f4_3.size())
        print(f5_3.size())
        print(ffc7.size())
        print(f6_2.size())
        print(f7_2.size())

        f3_3 = f3_3.view(-1, 256*16*16)
        f4_3 = f4_3.view(-1, 512*8*8)
        f5_3 = f5_3.view(-1, 512*4*4)
        ffc7 = ffc7.view(-1, 1024*6*6)
        f6_2 = f6_2.view(-1, 512*3*3)
        f7_2 = f7_2.view(-1, 256*2*2)

        h1 = F.relu(self.biLin1(f3_3, f4_3))
        h2 = F.relu(self.biLin2(f5_3, ffc7))
        h3 = F.relu(self.biLin3(f6_2, f7_2))

        h4 = F.relu(self.biLin4(h1, h2))
        h5 = F.relu(self.biLin5(h4, h3))

        h = F.relu(self.features(h5))
        h = F.log_softmax(self.classifier(h), 1)

        # h = h.view(-1, 256*2*2)

        # h = F.relu(self.features(h))
        # h = F.log_softmax(self.classifier(h), 1)







        # f3_3 = self.conv3_3_norm(f3_3)
        # f4_3 = self.conv4_3_norm(f4_3)
        # f5_3 = self.conv5_3_norm(f5_3)

        # cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        # reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        # cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        # reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        # cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        # reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        # cls4 = self.fc7_mbox_conf(ffc7)
        # reg4 = self.fc7_mbox_loc(ffc7)
        # cls5 = self.conv6_2_mbox_conf(f6_2)
        # reg5 = self.conv6_2_mbox_loc(f6_2)
        # cls6 = self.conv7_2_mbox_conf(f7_2)
        # reg6 = self.conv7_2_mbox_loc(f7_2)

        # # max-out background label
        # chunk = torch.chunk(cls1,4,1)
        # bmax  = torch.max(torch.max(chunk[0],chunk[1]),chunk[2])
        # cls1  = torch.cat([bmax,chunk[3]],dim=1)

        # return [cls1,reg1,cls2,reg2,cls3,reg3,cls4,reg4,cls5,reg5,cls6,reg6]

        return h

    def getEncoding(self, input):

        h = F.relu(self.conv1_1(input))
        h = F.relu(self.conv1_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h)); f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h)); f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h)); f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h));     ffc7 = h
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h)); f6_2 = h
        h = F.relu(self.conv7_1(h))
        h = F.relu(self.conv7_2(h)); f7_2 = h

        f3_3 = f3_3.view(-1, 256*16*16)
        f4_3 = f4_3.view(-1, 512*8*8)
        f5_3 = f5_3.view(-1, 512*4*4)
        ffc7 = ffc7.view(-1, 1024*6*6)
        f6_2 = f6_2.view(-1, 512*3*3)
        f7_2 = f7_2.view(-1, 256*2*2)

        h1 = F.relu(self.biLin1(f3_3, f4_3))
        h2 = F.relu(self.biLin2(f5_3, ffc7))
        h3 = F.relu(self.biLin3(f6_2, f7_2))

        h4 = F.relu(self.biLin4(h1, h2))
        h5 = F.relu(self.biLin5(h4, h3))

        encoding = self.features(h5)

        # h = h.view(-1, 256*2*2)

        # encoding = self.features(h)

        return encoding

    def isSame(self, input1, input2):

        enc1 = self.getEncoding(input1)
        enc2 = self.getEncoding(input2)

        return torch.dist(enc1, enc2)



