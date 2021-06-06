

"""# Test"""

from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models
import numpy as np
    
# Create list of all targets (minus background noise)
#dataset_path = '/Users/bartlomiejgasyna/Downloads/kakapo_model/data_speech_commands_v0-1.02'
dataset_path = '/Users/bartlomiejgasyna/Downloads/kakapo_model/kakapo_copy/dzwiek'
all_targets = all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
#all_targets.remove('_background_noise_')
#all_targets.remove('_noise')
#all_targets.remove('kakapo')
#all_targets.remove('kakapo_new')
print(all_targets)

# Settings
feature_sets_path = '/Users/bartlomiejgasyna/Downloads/kakapo_model/kakapo_copy'
feature_sets_filename = 'train_set_with_kakapo_bezszumu.npz'
model_filename = 'kakpo_without_noise.h5'
wake_word = 'kakapo_without_noise'



czy_model = 1 # Jeżeli chcemy stworzyć nowy model - 1, jeśli testujemy istniejący model - 0
if (not czy_model): 
#    feature_sets_filename = 'test_set_with_noise.npz'
    feature_sets_filename = 'test_set_with_kakapo_bezszumu.npz' 
    wake_word = 'kakapo_new'
    

#/Users/bartlomiejgasyna/Downloads/kakapo_model/all_targets_mfcc_sets.npz 

# Load feature sets
feature_sets = np.load(join(feature_sets_path, feature_sets_filename))

#feature_sets_second = np.load(join(feature_sets_path, feature_sets_filename_test))
print(feature_sets.files)

# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']

x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

# Look at tensor dimensions
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# Peek at labels
print(y_val)

# Convert ground truth arrays to one wake word (1) and 'other' (0)
wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')

## Peek at labels after conversion
#print(y_val)
#
## What percentage of wake word appear in validation labels
#print(sum(y_val) / len(y_val))
#print(1 - sum(y_val) / len(y_val))

# View the dimensions of our input data
print(x_train.shape)

# CNN for TF expects (batch, height, width, channels)
# So we reshape the input tensors with a "color" channel of 1
if (czy_model == 1):
    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1],
                              x_train.shape[2],
                              1)
    x_val = x_val.reshape(x_val.shape[0],
                          x_val.shape[1],
                          x_val.shape[2],
                          1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)
#print(x_train.shape)
#print(x_val.shape)
print(x_test.shape)

# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]
print(sample_shape)

if (czy_model == 1):
            
    # Build model
    # Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
    model = models.Sequential()
    model.add(layers.Conv2D(16,
                            (2, 2),
                            activation='relu',
                            input_shape=sample_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    
    
    model.add(layers.Conv2D(16, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Display model
    model.summary()
    
    # Add training parameters to model
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    # Train
    history = model.fit(x_train,
                        y_train,
                        epochs=30,
                        batch_size=200,
                        validation_data=(x_val, y_val))
    
    # best - epochs = 10, batch_size=5000
    
    # Plot results
    import matplotlib.pyplot as plt
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
    # Save the model as a file
    models.save_model(model, model_filename)

    # See which are 'stop'
    for idx, y in enumerate(y_test):
        if y == 1:
            print(idx)
# Evaluate model with test set
model = models.load_model(model_filename)
model.evaluate(x=x_test, y=y_test)

if (not czy_model):
        # TEST: Load model and run it against test set
    model = models.load_model(model_filename)
    for i in range(1, len(y_test)):
        print('Answer:', y_test[i], ' Prediction:','0' if (model.predict(np.expand_dims(x_test[i], 0)) <0.5) else '1', model.predict(np.expand_dims(x_test[i], 0)))


