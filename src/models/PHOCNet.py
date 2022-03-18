from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import losses
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow_addons.layers import SpatialPyramidPooling2D
import numpy as np
from phoc_label_generator import phoc_generate_label
from imageio import imread
import pandas as pd
import math
import os
import tensorflow
import sys

#old_stdout = sys.stdout
#log_file = open("message.log","w")
#sys.stdout = log_file

def base_model(img_width, img_height, weight_path=None):
    if K.image_data_format() == 'channels_first':
        input_shapes = (3, img_width, img_height)
    else:
        input_shapes = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu', input_shape=input_shapes))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(SpatialPyramidPooling2D([1, 2, 4]))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(830, activation='sigmoid'))
    from tensorflow.keras.optimizers import SGD, Adam, Adadelta

    loss = losses.binary_crossentropy
    optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=5e-5)
    model.compile(loss=loss, optimizer=optimizer, metrics=[tensorflow.keras.metrics.CosineSimilarity(axis=1)])
    if weight_path:
        df = pd.read_pickle(weight_path)
        tmp_weights = df.values
        N = len(tmp_weights)
        weights = []
        for i in range(N):
            weights.append(tmp_weights[i][0])
        model.set_weights(weights)

    model.summary()
    from keras.utils.vis_utils import plot_model as plot

    plot(model, to_file="phocnet.png", show_shapes=True)
    return model

def map(model, x_test, y_test, transcripts):
  """This module evaluates the partially trained model using Test Data
  Args:
    model: Instance of Sequential Class storing Neural Network
    x_test: Numpy storing the Test Images
    y_test: Numpy storing the PHOC Labels of Test Data
    transcripts: String storing the characters in the Image.
  Returns:
    map: Floating number storing the Mean Average Precision.
  """
  y_pred = model.predict(x_test)
  y_pred = np.where(y_pred<0.5, 0, 1)
  N = len(transcripts)
  precision = {}
  count = {}
  for i in range(N):
    if transcripts[i] not in precision.keys():
      precision[transcripts[i]] = 1
      count[transcripts[i]] = 0
    else:
      precision[transcripts[i]] += 1

  for i in range(N):
    pred = y_pred[i]
    acc = np.sum(abs(y_test-pred), axis=1)
    tmp = np.argmin(acc)
    if transcripts[tmp] == transcripts[i]:
      count[transcripts[tmp]] += 1

  mean_avg_prec = [0, 0]
  for i in range(N):
    if precision[transcripts[i]] <= 1:
      continue
    mean_avg_prec[0] += count[transcripts[i]]*1.0/precision[transcripts[i]]
    mean_avg_prec[1] += 1

  map = mean_avg_prec[0]*1./mean_avg_prec[1]
  print ("The Mean Average Precision = ", map)
  print ("Total test cases = ", N)
  return map

def get_generator_value(class_indicates, index):
    key_list = list(class_indicates.keys())
    val_list = list(class_indicates.values())
    return key_list[val_list.index(index)]

train_path = 'asar-dataset/train'
test_path = 'asar-dataset/test'
val_path = 'asar-dataset/val'

train_datagen = ImageDataGenerator()

#val_datagen = ImageDataGenerator(rescale=1. / 255.)
val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_path,
        shuffle= False,
        # All images will be resized to 150x150
        target_size=(16, 16),
        batch_size=1,
        # binary: use binary_crossentropy loss, we need binary labels
        # categorical : use categorical_crossentropy loss, then need categorical labels
        class_mode='binary')

val_generator = val_datagen.flow_from_directory(
        # This is the target directory
        val_path,
        shuffle= False,
        target_size=(16, 16),
        batch_size=1,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_path,
        shuffle= False,
        target_size=(16, 16),
        batch_size=1,
        class_mode='binary')

train_generator.reset()
y_train = train_generator.labels
X_train = [np.array(tensorflow.image.resize(imread(train_path + '/' + file), [110,110])) for file in train_generator.filenames]
X_train = np.array(X_train)

val_generator.reset()
y_val = val_generator.labels
X_val = [np.array(tensorflow.image.resize(imread(val_path + '/' + file), [110,110])) for file in val_generator.filenames]
X_val = np.array(X_val)

test_generator.reset()
y_test = test_generator.labels
X_test = [np.array(tensorflow.image.resize(imread(test_path + '/' + file), [110,110])) for file in test_generator.filenames]
X_test = np.array(X_test)

print('Data has been loaded')

# num_of_classes = len(train_generator.class_indices)
test_transcripts = [get_generator_value(test_generator.class_indices, int(i)) for i in y_test]
test_transcripts = np.array(test_transcripts)
y_train = [phoc_generate_label(get_generator_value(train_generator.class_indices, int(i))) for i in y_train]
y_train = np.array(y_train)
y_val = [phoc_generate_label(get_generator_value(val_generator.class_indices, int(i))) for i in y_val]
y_val = np.array(y_val)
y_test = [phoc_generate_label(get_generator_value(test_generator.class_indices, int(i))) for i in y_test]
y_test = np.array(y_test)

weight_path = 'phoc_weights.pkl'
if os.path.exists(weight_path):
    model = base_model(110, 110, weight_path= weight_path)
else:
    model = base_model(110, 110)
batch_size = 32

map_max = 0

for i in range(5):
    history = model.fit(
        X_train, y_train,
        steps_per_epoch=math.ceil(train_generator.samples//batch_size),
        batch_size=batch_size,
        epochs=5,
        shuffle= True,
        validation_data=(X_val, y_val),
        validation_steps=math.ceil(val_generator.samples//batch_size),
        verbose=1)

    map_value = map(model, X_test, y_test, test_transcripts) # Calculates the MAP of the model
    # save the model
    if map_value > map_max:
        map_max = map_value
        weights = model.get_weights()
        df = pd.DataFrame(weights)
        print("Saving the best model.......")
        model.save('phoc-model.h5')
        df.to_pickle('phoc_weights.pkl')
#
# # Create directory to store training history
#
# if not os.path.exists("Train_History"):
#     os.makedirs("Train_History")
#
# # Store train history as CSV file
# model_name="phoc-model"
# hist_df = pd.DataFrame(history.history)
# hist_csv_file = 'Train_History/history_'+model_name+'.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
# # Plot train and validation accuracy(avg cosine similarity)
#
# acc = history.history['cosine_similarity']
# val_acc = history.history['val_cosine_similarity']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc,label='Training Similarity')
# plt.plot(epochs, val_acc,label='Validation Similarity')
# plt.title(model_name+'_Cosine Similarity')
# plt.legend()
# plt.savefig('Train_History/'+model_name+'_Pretrain_CS.png')
# plt.show()
#
# # Plot train and validation loss
# plt.plot(epochs, loss,label='Training Loss')
# plt.plot(epochs, val_loss,label='Validation Loss')
# plt.title(model_name+' MSE Loss')
# plt.legend()
# plt.savefig('Train_History/'+model_name+'_Pretrain_Loss.png')
# plt.show()

#sys.stdout = old_stdout
#log_file.close()