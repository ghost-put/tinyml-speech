#!/usr/bin/env python
# coding: utf-8

# # ***Biblioteki, wczytanie danych***

# In[28]:


from os import listdir
from os.path import isdir, join
from os import remove
import librosa
import random

import numpy as np
import matplotlib.pyplot as plt
get_ipython().system('pip install python_speech_features')
import python_speech_features


# In[34]:


#@title Default title text

# # # Dataset path and view possible targets
# from google.colab import drive
# drive.mount('/content/drive')

# dataset_path = '/content/drive/MyDrive/data_speech_commands_v0.02'
dataset_path = '/Users/bartlomiejgasyna/Downloads/kakapo_model/kakapo_copy/dzwiek'
#dataset_path = '/Users/bartlomiejgasyna/Downloads/kakapo_model/data_speech_commands_v0-1.02'
for name in listdir(dataset_path):
    if isdir(join(dataset_path, name)):
        print(name)


# In[35]:



# Create an all targets list
all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
print(all_targets)


# In[36]:


# Leave off background noise set
#all_targets.remove('_background_noise_')

all_targets.remove('kakapo_new')
all_targets.remove('_noise')
all_targets.remove('kakapo')
all_targets.remove('random')
all_targets.remove('noise2')
#all_targets.remove('kakapo_without_noise')


print(all_targets)


# In[37]:



# See how many files are in each
num_samples = 0
for target in all_targets:
    print(len(listdir(join(dataset_path, target))))
    num_samples += len(listdir(join(dataset_path, target)))
print('Total samples:', num_samples)


# In[38]:


# Settings
target_list = all_targets
feature_sets_file = 'test_set_with_kakapo_bezszumu.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.0
test_ratio = 1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16


# In[39]:


# Create list of filenames along with ground truth vector (y)
filenames = []
y = []
for index, target in enumerate(target_list):
    print(join(dataset_path, target))
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


# In[40]:


print(y)
for item in y:
    print(len(item))


# In[41]:


# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]


# In[42]:


# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)


# In[43]:


# for i in y:
#   print(i)


# In[44]:


# Only keep the specified number of samples (shorter extraction/training)
print(len(filenames))
filenames = filenames[:int(len(filenames) * perc_keep_samples)]
print(len(filenames))


# In[45]:


# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)


# In[46]:


# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames
filenames_train = filenames[(val_set_size + test_set_size):]


# In[47]:


# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_test = y
y_orig_train = y[(val_set_size + test_set_size):]


# In[48]:


# Function: Create MFCC from given path
def calc_mfcc(path):

    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
                                            samplerate=fs,
                                            winlen=0.256,
                                            winstep=0.050,
                                            numcep=num_mfcc,
                                            nfilt=26,
                                            nfft=2048,
                                            preemph=0.0,
                                            ceplifter=0,
                                            appendEnergy=False,
                                            winfunc=np.hanning)
    return mfccs.transpose()


# In[49]:



# TEST: Construct test set by computing MFCC of each WAV file
prob_cnt = 0
x_test = []
y_test = []
filenames_to_remove = list(filenames_train)
y_to_remove = list(y)
for index, filename in enumerate(filenames_train):
    # print(index, filename)

    # Stop after 500
    if index >= 500:
        break

    # Create path from given filename and target item
    path = join(dataset_path, target_list[int(y_orig_train[index])],
                filename)

    # Create MFCCs
    mfccs = calc_mfcc(path)

    if mfccs.shape[1] == len_mfcc:
        x_test.append(mfccs)
        y_test.append(y_orig_train[index])
    else:
        print('Dropped:', index, filename, mfccs.shape, path)
        prob_cnt += 1

# filenames_train = tuple(filenames_to_remove)
# y = tuple(y_to_remove)

# # Break y apart into train, validation, and test sets
# y_orig_val = y[:val_set_size]
# y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
# y_orig_train = y[(val_set_size + test_set_size):]






# In[51]:


print('% of problematic samples:', prob_cnt / 500)


# In[52]:


# TEST: Test shorter MFCC
#get_ipython().system('pip install playsound')
#from playsound import playsound
#
#idx = 13
#
## Create path from given filename and target item
#path = join(dataset_path, target_list[int(y_orig_train[idx])],
#            filenames_train[idx])
#
## Create MFCCs
#mfccs = calc_mfcc(path)
#print("MFCCs:", mfccs)
#
## Plot MFCC
#fig = plt.figure()
#plt.imshow(mfccs, cmap='inferno', origin='lower')
#
## TEST: Play problem sounds
#print(target_list[int(y_orig_train[idx])])
# playsound(path)


# In[53]:


#print(y_orig_train[idx])


# In[54]:


# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        # Create path from given filename and target item
        path = join(dataset_path, target_list[int(in_y[index])],
                    filename)

        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1
#            remove(path)

    return out_x, out_y, prob_cnt


# In[55]:


# Create train, validation, and test sets
x_train, y_train, prob = extract_features(filenames_train,
                                          y_orig_train)
#print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
#print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
#print('Removed percentage:', prob / len(y_orig_test))


# In[66]:


print(len(y_orig_train))
print(filenames_train)


# In[63]:


#type(np.load(feature_sets_file))


# In[65]:





# In[59]:


# TEST: Load features
#feature_sets = np.load(feature_sets_file)
#feature_sets.files



# In[58]:



# Save features and truth vector (y) sets to disk'

np.savez(feature_sets_file,
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)


# In[61]:


#len(feature_sets['x_train'])
#print(feature_sets['y_val'])
