import pickle
from glob import glob
import os
import pandas as pd
import csv
from csv import writer
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import splitfolders


#### Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--path_to_joints',metavar='path_to_joints',  default='', type=str, help='Path of the joints set')
parser.add_argument('--path_to_images',metavar='path_to_img',  default='', type=str, help='Path of the images set')
parser.add_argument('--csv_name',metavar='csv_name',  default='csv_name', type=str, help='name of the csv ')
parser.add_argument('--threshold',metavar='threshold',   type=float, help='value for threshold ')
parser.add_argument('--dataset_name',metavar='dataset_name',default='new', type=str, help=' dataset_name')

args= parser.parse_args()

path_joint = args.path_to_joints
path_img = args.path_to_images
csv_name = args.csv_name
threshold = args.threshold
dataset_name = args.dataset_name



if not os.path.exists(dataset_name):
    print("It doesnt exist! Lets create it ")
    os.makedirs(dataset_name)


#Creating list for scans and images#


scans = glob(path_joint + '*pkl')
scans.sort()

images =  glob(path_img + '*jpg')
images_list = []



# laser part 
#lasers = glob(path_to_laser + '*pkl')
# lasers.sort()
# 
# for l in lasers:
#     #print(l)
#     file_to_read = open(l, "rb")
#     loaded_dictionary_lasers = pickle.load(file_to_read)
#     print(loaded_dictionary_lasers)
#     #print(loaded_dictionary_lasers.keys())
# 
#     ranges = loaded_dictionary_lasers["ranges"]
#     ranges_reshaped = np.array(ranges).reshape(1, 1081)
# 
#     ranges_200_400 = ranges_reshaped[0][200:400]
# 
#     if np.any(ranges_200_400 <= 1.2):
#         print("")


for i in images:

    output_filename_image = os.path.basename(i).split('.')[0]
    images_list.append(output_filename_image)
    images_list.sort()
    #print(images_list)


output_path = "%s" % dataset_name + os.sep


limit= 1.0

def label_maker(fn):   ### for hearacleia

    if  (fn[0]<= threshold and fn[0]> -threshold) and (fn[1]<= threshold and fn[1]> -threshold) and (fn[2]<= threshold and fn[2]> -threshold) and (fn[3]<= 0.01 and fn[3]> -threshold):
        return 0

    if  (fn[0]<= -threshold ) and (fn[1]<= -threshold) and (fn[2]<= -threshold) and (fn[3]<= -threshold):
        return 0

    if  (fn[0]> threshold+limit ) and (fn[1]> threshold+limit) and (fn[2]> threshold+limit) and (fn[3]> threshold+limit):

        return 1

    else:
        return 2


velocity_array=np.empty([1,4])
csv_final = output_path + csv_name+ '.csv'

with open(csv_final,'w') as final_file:
    for s in scans:
        file_to_read = open(s, "rb")
        loaded_dictionary = pickle.load(file_to_read)
        velocity =loaded_dictionary["velocity"]

        vel_out = label_maker(velocity)

        tmp=np.array(velocity).reshape(1,4)
        velocity_array=np.append(velocity_array, tmp,axis=0)

        output_filename = os.path.basename(s).split('.')[0]
        #output_filename_image = os.path.basename(images).split('.')[0]
        final_file.write(output_filename)
        #final_file.write(images_list[])
        final_file.write('\t' + ',' + str(vel_out) + '\n')
        #final_file.write(str(images_list))


for i in range(0,4):
    plt.figure("Velocity for wheel"+ str(i))
    plt.hist(velocity_array[:,i], bins='auto')  # arguments are passed to np.histogram


#plt.show()


final_file.close()

df = pd.read_csv(csv_final, header = None)
headers = ['joint', 'label', 'image_fn']

### appending the images
df[''] = images_list

df.to_csv(output_path + csv_name+ '_final'+ '.csv', index=False, header=headers)


#Creating the folders for positive and negative
positive_set= 'positive_set_' + dataset_name + os.sep
negative_set= 'negative_set_' + dataset_name+ os.sep
positive_path = os.path.join(dataset_name, positive_set)
negative_path = os.path.join(dataset_name, negative_set)

if not os.path.exists(positive_set):
    print("Positive set is to be created")

    os.makedirs(positive_path,exist_ok=True)


if not os.path.exists(negative_set):
    print("negative set is to be created")
    #shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(negative_path, exist_ok=True)


df = pd.read_csv(dataset_name+ '/' +csv_name + '_final' + '.csv' )

positive = df[df["label"] == 1]
list_positive = (positive['image_fn']).tolist()

print(len(list_positive))

negative = df[df["label"] == 0]
list_negative = (negative['image_fn']).tolist()
print(len(list_negative))

images_list = [eval(i) for i in images_list]
print(len(images_list))

source_folder = path_img
destination_folder_positive = positive_path
destination_folder_negative = negative_path

##### Fill the folders with positive and negative data ######
for i in images_list:
    source = source_folder + str(i) + '.jpg'
    #destination = destination_folder

    if i in list_positive:
        shutil.copy(source,destination_folder_positive)
    elif i in list_negative:
        shutil.copy(source,destination_folder_negative)
    #else:
        #print("Undecided")


#Creating training and test dataset 

input_folder = dataset_name
splitfolders.ratio(input_folder, output=dataset_name,
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False) # default values
