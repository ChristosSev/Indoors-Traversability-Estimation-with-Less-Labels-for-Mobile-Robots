from transformers import ViTFeatureExtractor
import requests
from PIL import Image
import numpy as np
import torch
import glob
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import copy
from transformers import ViTMAEForPreTraining
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import seaborn as sns
import pandas as pd
import os
from sklearn.manifold import TSNE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


outDir = "/home/christos/results/"python


path_to_train = '/home/christos/data/liga/erb'
path_to_test = '/home/christos/data/liga/nh/train'

print(path_to_train)
print(path_to_test)


num_epochs=50

def showTSNE(x, y, epoch):
#     x = np.array(x)
#     y = np.array(y)
    #print(x.size(), y.size())
    tsne = TSNE(n_components = 2)
    tsneResults = tsne.fit_transform(x)
    tsneResultsDF = pd.DataFrame({'tsne1': tsneResults[:,0], 'tsne2': tsneResults[:,1], 'label': y})
    outDir = "/home/christos/results/"
    fig, ax = plt.subplots(1)
    sns.scatterplot(x = 'tsne1', y = 'tsne2', hue = 'label', data = tsneResultsDF, ax = ax, s = 120).set(xlabel=None,ylabel=None) 
    lim = (tsneResults.min() - 5, tsneResults.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    handles, labels  =  ax.get_legend_handles_labels()
    ax.legend(handles, ['Positive', 'Negative'], loc='upper right')
#     ax.legend(['Yes', 'No'], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.0)
    plt.savefig(os.path.join(outDir, '{:05d}_ch.png'.format(epoch)))
    plt.close()
class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        #self.nf = 768

        self.fc1 = nn.Sequential(
            nn.Linear(768, 384),
            #nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )

   

        #self.mha = torch.nn.MultiheadAttention(384, 1, batch_first=True)
	#self.mha = torch.nn.MultiheadAttention(96, 1, batch_first=True)

	
        self.fc = nn.Sequential(
            nn.Linear(384
, 2),
        )


    def forward(self, x):
        x = self.fc1(x)
      #  x = self.fc2(x)

       	#x, _ = self.mha(x,x,x)
        x = self.fc(x)
        return x


#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Transforms
transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])



train_loader=DataLoader(
    torchvision.datasets.ImageFolder(path_to_train,transform=transformer),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(path_to_test,transform=transformer),
    batch_size=64, shuffle=False
)

#calculating the size of training and testing image
train_count=len(glob.glob(path_to_train+'/**/*.jpg'))
test_count=len(glob.glob(path_to_test+'/**/*.jpg'))



transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)


from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
model.eval()


proj = Projector().to(device)

model.classifier = nn.Linear(768, 768)

model.to(device)

i = 0

#comment for unfreeze
for name, param in model.named_parameters():
    print(i, name)
    if i < 197:
        param.requires_grad = False
    i += 1

allWeights = list(model.parameters()) + list(proj.parameters())
optimizer= SGD(allWeights, lr=0.01, weight_decay=0.0001)
#optimizer = Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

loss_function=nn.CrossEntropyLoss()

#Model training and saving best model


best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        #print(i)
        if torch.cuda.is_available():
            #print('bbbb')
            images = Variable(images.cuda())
            #print(images)
            labels = Variable(labels.cuda())
            #print(labels)
        optimizer.zero_grad()



        # print(outputs.size())
        outputs = model(images).logits
        # print(outputs.size())
        outputs = proj(outputs)
        #print(outputs.size())
        loss = loss_function(outputs, labels)
        #
        loss.backward()
        optimizer.step()
      
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs, 1)

        train_accuracy += float(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    labels_list = []
    prediction_list = []
    x_list = []
    y_list = []

    for i, (images, labels) in enumerate(test_loader):
        
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            labels_list.extend((labels.cpu()))

        outputs = model(images).logits
        x_list.append(outputs)
        y_list.append(labels)
        outputs = proj(outputs)
        _, prediction = torch.max(outputs, 1)
        prediction_list.extend(prediction.cpu())
        # print(prediction_list)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count


    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))
    x_tensor = torch.cat(x_list, dim = 0)
    y_tensor = torch.cat(y_list, dim = 0)
    showTSNE(x_tensor.cpu().detach(), y_tensor.cpu().detach(), epoch)

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy
    

#For confusion matrix 
#     classes = ('Positive' , 'Negative')
#     cf_matrix = confusion_matrix(labels_list, prediction_list)
#     df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 1, index=[i for i in classes],
#                          columns=[i for i in classes])
#     plt.figure(figsize=(12, 7))
#     sns.heatmap(df_cm, annot=True)
#     plt.savefig(os.path.join(outDir, '{:05d}_cm_wh_least_vit.png'.format(epoch)))
#     plt.close()


labels_list = np.array(labels_list)
prediction_list = np.array(prediction_list)



# cm = confusion_matrix(labels_list,prediction_list)
# print(cm)


print(path_to_train)
print(path_to_test)

