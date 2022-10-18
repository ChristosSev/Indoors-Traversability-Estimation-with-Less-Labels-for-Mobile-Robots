#from transformers import ViTFeatureExtractor
import requests
from PIL import Image
import numpy as np
import torch
import glob
from sklearn.manifold import TSNE
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
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from transformers import ViTMAEForPreTraining
# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor

#import lightly


path_to_train = '/home/christos_sevastopoulos/Desktop/friaksimo/liga/nh/train'
path_to_test = '/home/christos_sevastopoulos/Desktop/friaksimo/liga/nh/train'
num_epochs=1

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
    batch_size=16, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(path_to_test,transform=transformer),
    batch_size=16, shuffle=False
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


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 64, 3, 2, 1)
      self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
      self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
      self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)


      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(2048, 128)
      self.fc2 = nn.Linear(128, 2)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      #print(x.size())
      x = self.conv1(x)
      #print(x.size())
      # Use the rectified-linear activation function over x
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
     # print(x.size())

      x = self.conv2(x)
      x = F.relu(x)
      #print(x.size())
      # Run max pooling over x
      x = F.max_pool2d(x, 2)
    

      x = self.conv3(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
   

      x = self.conv4(x)
      x = F.relu(x)
     # print(x.size())
    
      x = torch.flatten(x, 1)
      # Pass data through fc1



      x = self.fc1(x)
      x = F.relu(x)
      # x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output



model = Net()
 
model.eval()


model.to(device)

optimizer= SGD(model.parameters(),lr=0.001,weight_decay=0.0001)


loss_function=nn.CrossEntropyLoss()

best_accuracy = 0.0

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        #print(i)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
      
        tsne = TSNE(n_components=2).fit_transform(outputs.cpu().detach().numpy())
        #print("alekos")
       # print(tsne)
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

#classes = ('Positive' , 'Negative')
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

labels_list = np.array(labels_list)
prediction_list = np.array(prediction_list)

tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)



fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for label in labels_list:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(labels) if l == label]

    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)

    # convert the class color to matplotlib format
    color = np.array(labels_list[label], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()

