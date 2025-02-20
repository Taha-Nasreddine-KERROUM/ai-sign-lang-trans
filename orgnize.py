import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


sdir=r'C:\Users\PC\Desktop\data\processed_combine_asl_dataset'
classlist=sorted(os.listdir(sdir))
print (classlist)
filepaths = []
labels=[]
for klass in classlist:
    classpath=os.path.join(sdir, klass)
    flist=sorted(os.listdir(classpath))
    for f in flist:
        fpath=os.path.join(classpath,f)
        filepaths.append(fpath)
        labels.append(klass)
Fseries=pd.Series(filepaths, name='filepaths')
Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries, Lseries], axis=1)
train_df, dummy_df=train_test_split(df, train_size=.95, shuffle=True, random_state=123, stratify=df['labels'])
valid_df, test_df= train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])
print('train_df lenght: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
# get the number of classes and the images count for each class in train_df
classes=sorted(list(train_df['labels'].unique()))
class_count = len(classes)
print('The number of classes in the dataset is: ', class_count)
groups=train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
countlist=[]
classlist=[]
for label in sorted(list(train_df['labels'].unique())):
    group=groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

# get the classes with the minimum and maximum number of train images
max_value=np.max(countlist)
max_index=countlist.index(max_value)
max_class=classlist[max_index]
min_value=np.min(countlist)
min_index=countlist.index(min_value)
min_class=classlist[min_index]
print(max_class, ' has the most images= ',max_value, ' ', min_class, ' has the least images= ', min_value)
# lets get the average height and width of a sample of the train images
ht=0
wt=0
# select 100 random samples of train_df
train_df_sample=train_df.sample(n=100, random_state=123,axis=0)
for i in range (len(train_df_sample)):
    fpath=train_df_sample['filepaths'].iloc[i]
    img=plt.imread(fpath)
    shape=img.shape
    ht += shape[0]
    wt += shape[1]
print('average height= ', ht//100, ' average width= ', wt//100, 'aspect ratio= ', ht/wt)

def trim(df, max_samples, min_samples, column):
    df=df.copy()
    groups=df.groupby(column)
    trimmed_df = pd.DataFrame(columns = df.columns)
    groups=df.groupby(column)
    for label in df[column].unique():
        group=groups.get_group(label)
        count=len(group)
        if count > max_samples:
            sampled_group=group.sample(n=max_samples, random_state=123,axis=0)
            trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if count>=min_samples:
                sampled_group=group
                trimmed_df=pd.concat([trimmed_df, sampled_group], axis=0)
    print('after trimming, the maximum samples in any class is now ',max_samples, ' and the minimum samples in any class is ', min_samples)
    return trimmed_df

max_samples=250
min_samples=0
column='labels'
train_df=trim(train_df, max_samples, min_samples, column)


def balance(df, n, working_dir, img_size):
    def augment(df, n, working_dir, img_size):
        aug_dir = os.path.join(working_dir, 'aug')
        os.mkdir(aug_dir)
        for label in df['labels'].unique():
            dir_path = os.path.join(aug_dir, label)
            os.mkdir(dir_path)
        # create and store the augmented images
        total = 0
        gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                                 height_shift_range=.2, zoom_range=.2)
        groups = df.groupby('labels')  # group by class
        for label in df['labels'].unique():  # for every class
            group = groups.get_group(label)  # a dataframe holding only rows with the specified label
            sample_count = len(group)  # determine how many samples there are in this class
            if sample_count < n:  # if the class has less than target number of images
                aug_img_count = 0
                delta = n - sample_count  # number of augmented images to create
                target_dir = os.path.join(aug_dir, label)  # define where to write the images
                msg = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(delta))
                print(msg, '\r', end='')  # prints over on the same line
                aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=img_size,
                                                  class_mode=None, batch_size=1, shuffle=False,
                                                  save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                                  save_format='jpg')
                while aug_img_count < delta:
                    images = next(aug_gen)
                    aug_img_count += len(images)
                total += aug_img_count
        print('Total Augmented images created= ', total)
        # create aug_df and merge with train_df to create composite training set ndf
        aug_fpaths = []
        aug_labels = []
        classlist = os.listdir(aug_dir)
        for klass in classlist:
            classpath = os.path.join(aug_dir, klass)
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries = pd.Series(aug_fpaths, name='filepaths')
        Lseries = pd.Series(aug_labels, name='labels')
        aug_df = pd.concat([Fseries, Lseries], axis=1)
        df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
        return df

    df = df.copy()
    # make directories to store augmented images
    aug_dir = os.path.join(working_dir, 'aug')
    if 'aug' in os.listdir(working_dir):
        print(
            ' Augmented images already exist. To delete these and create new images enter D, else enter U to use these images',
            flush=True)
        ans = input(' ')
        if ans == 'D' or ans == 'd':
            shutil.rmtree(aug_dir)  # start with an clean empty directory
            augment(df, n, working_dir, img_size)
            return df
        else:

            return df
    else:
        augment(df, n, working_dir, img_size)
        return df


n = 200  # number of samples in each class
working_dir = r'./'  # directory to store augmented images
img_size = (200, 200)  # size of augmented images
train_df = balance(train_df, n, working_dir, img_size)