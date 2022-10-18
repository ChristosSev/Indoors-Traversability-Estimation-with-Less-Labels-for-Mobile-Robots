import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.autograd import Variable
import torchvision
import pathlib
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
import seaborn as sns
import pandas as pd

num_epochs=50
outDir = "/home/christos/results/"
#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Transforms
transformer=transforms.Compose([
   transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

train_path = '/home/christos/data/liga/woh/train'
test_path= '/home/christos/data/liga/nh/train'



print(train_path)
print(test_path)

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=16, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=16, shuffle=False
)




class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        #self.nf = 768

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            #nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )

   

        #self.mha = torch.nn.MultiheadAttention(384, 1, batch_first=True)
	#self.mha = torch.nn.MultiheadAttention(96, 1, batch_first=True)

	
        self.fc = nn.Sequential(
            nn.Linear(1024
, 2),
        )


    def forward(self, x):
        x = self.fc1(x)
      #  x = self.fc2(x)

       	#x, _ = self.mha(x,x,x)
        x = self.fc(x)
        return x






#calculating the size of training and testing image
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

print(train_count,test_count)

#define the model

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
print(num_ftrs)
model.fc = nn.Linear(num_ftrs, 2048)

model.to(device)
proj = Projector().to(device)


allWeights = list(model.parameters()) + list(proj.parameters())
optimizer= SGD(allWeights,lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

# Model training and saving best model

best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += float(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    labels_list = []
    prediction_list = []


    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            labels_list.extend((labels.cpu()))

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        prediction_list.extend(prediction.cpu())
        #print(prediction)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count


    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))


    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
    
    classes = ('Positive' , 'Negative')
    cf_matrix = confusion_matrix(labels_list, prediction_list)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 1, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(outDir, '{:05d}_cm_uc.png'.format(epoch)))
    plt.close()

labels_list = np.array(labels_list)
prediction_list = np.array(prediction_list)

cm = confusion_matrix(labels_list,prediction_list)
print(cm)


