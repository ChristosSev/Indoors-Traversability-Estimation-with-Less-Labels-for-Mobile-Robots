import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot
from torchvision import models
from torchsummary import summary
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

#Define the parameters
ndf = ngf = 64
num_epochs = 75
image_size = 64
ngpu = 1
nz = 100
nc = 3
batch_size = 64
lr = 0.0001
beta1 = 0.85

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#DIrectories
dataroot = "/home/christos/data/COLOR_OLA/"

train_data_directory = "/home/christos/data/liga/Mix/UC_NH_WH_ERB/train"
test_data_directory = '/home/christos/data/liga/ELB/train'


#Access the data
train_data = dset.ImageFolder(train_data_directory, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
test_data = dset.ImageFolder(test_data_directory, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))

# print(train_data.class_to_idx)
# print(test_data.class_to_idx)

#Load the data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64

            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

        )
        self.linear2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1),
            nn.Sigmoid()
        )
        # nn.Linear(4*4*512,2)

    def forward(self, x):
        # print(x.size())
        x = self.main(x)
        # print(x.size())
        x = x.view(-1,512*4*4)
        # print(x.size())
        x = self.linear2(x)
        # print(x.size())
        x = x.view(-1)
        # print(x.size())
        return x
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # input is Z, going into a convolution
        # self.linear1 = nn.Linear(nz,512*8*8)
        self.linear1 = nn.Sequential(
            nn.Linear(nz, 512 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8,ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
        )
    def forward(self, x):
        x = self.linear1(x)
        x = x.view(-1,512, 4, 4)
        x = self.deconv1(x)
        return x
class InvG(nn.Module):
    def __init__(self, ngpu):
        super(InvG, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
    )
        self.linear2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.Sigmoid()
    )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,512*4*4)
        x = self.linear2(x)

        return x

class FL(nn.Module):
    def __init__(self, ngpu):
        super(FL, self).__init__()
        self.ngpu = ngpu
        # input is Z, going into a convolution
        # self.linear1 = nn.Linear(nz,512*8*8)
        self.fc1 = nn.Sequential(
            nn.Linear(12288, 1))
            # nn.ReLU(True))

        self.fc2 = nn.Sequential(
            nn.Linear(8192,1))
            # nn.ReLU(True))

        self.fc3 = nn.Sequential(
            nn.Linear(8192, 1))
            # nn.ReLU(True))
        self.fc4 = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2, x_3):
        x_1 = self.fc1(x_1)

        #x = x.view(-1, 512, 4, 4)
        x_2 = self.fc2(x_2)

        x_3 = self.fc3(x_3)
        c = torch.cat((x_1,x_2,x_3),1)
        x = self.fc4(c)
        return x


netL = FL(ngpu)
netL.to(device)


netD = Discriminator(ngpu)
netG = Generator(ngpu)
netI = InvG(ngpu)
netD.load_state_dict(torch.load('model269D.pth'))
netG.load_state_dict(torch.load('model269H.pth'))
netI.load_state_dict(torch.load('model269I.pth'))

netD.eval()
netG.eval()
netI.eval()


#Extract the features
netD = nn.Sequential(*list(netD.children())[:1])

netD.to(device)
netG.to(device)
netI.to(device)


netD.requires_grad = False
netG.requires_grad = False
netI.requires_grad = False


optimizer = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

#Training
acc_list  = np.array([])
prec_list = np.array([])
rec_list  = np.array([])
for epoch in range(num_epochs):
   # print("Training...")

    for i, data in enumerate(train_loader, 0):
        netL.zero_grad()
        # print(data[1])
        image_real, label = data
        image_real = image_real.to(device)
        b_size = image_real.size(0)
        label = label.type(torch.float).to(device)
        image_inv = netI(image_real)
        image_fake = netG(image_inv)
        dis_real = netD(image_real)
        dis_fake = netD(image_fake)
        #features!!!
        resLoss = abs(image_real - image_fake).view(b_size,-1)
        disLoss = abs(dis_real - dis_fake).view(b_size,-1)
        disFeat = dis_real.view(b_size,-1)
        # print(resLoss.size())
        # print(disLoss.size())
        # print(disFeat.size())

        output = netL(resLoss, disLoss, disFeat).view(-1)

        err = criterion(output, label)
        # Calculate gradients for D in backward pass+
        err.backward()
        loss = output.mean().item()
        optimizer.step()

   # print("Testing...")
    y_pred_list = np.array([])
    y_test = np.array([])
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # print(data[1])
            image_real, label = data
            image_real = image_real.to(device)
            b_size = image_real.size(0)
            label = label.type(torch.float).to(device)
            image_inv = netI(image_real)
            image_fake = netG(image_inv)
            dis_real = netD(image_real)
            dis_fake = netD(image_fake)
            # features!!!
            resLoss = abs(image_real - image_fake).view(b_size, -1)
            disLoss = abs(dis_real - dis_fake).view(b_size, -1)
            disFeat = dis_real.view(b_size, -1)
            # print(resLoss.size())
            # print(disLoss.size())
            # print(disFeat.size())

            output = netL(resLoss, disLoss, disFeat).view(-1)

            err = criterion(output, label)
            # Calculate gradients for D in backward pass+
            loss = output.mean().item()
            # acc = binary_acc(output, label)

            y_pred_tag = torch.round(output)
            y_pred_list = np.hstack((y_pred_list, y_pred_tag.cpu()))
            y_test = np.hstack((y_test, label.cpu()))
            # Output Testing stats

    # print(y_pred_list)
    # print(y_test)
    # print(y_pred_list.shape)
    # print(y_test.shape)
    report = classification_report(y_test, y_pred_list, output_dict=True)
    acc = report["accuracy"] * 100
    print('[%d/%d]:\tAccuracy: %.4f' % (epoch+1, num_epochs, acc))
   # print(classification_report(y_test, y_pred_list))
    prec = report['1.0']['precision'] * 100
    rec  = report['1.0']['recall'] * 100
    acc_list  = np.hstack((acc_list, acc))
    prec_list = np.hstack((prec_list, prec))
    rec_list  = np.hstack((rec_list, rec))

plt.figure(figsize=(10,5))
plt.title("Stats During Training")
plt.plot(acc_list,label="A")
plt.plot(prec_list,label="P")
plt.plot(rec_list, label="R")
plt.xlabel("epoch")
plt.ylabel("%")
plt.legend()
plt.show()

