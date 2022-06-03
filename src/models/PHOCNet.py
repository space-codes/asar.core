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
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import random
import sys

#old_stdout = sys.stdout
#log_file = open("message.log","w")
#sys.stdout = log_file

# DataSequence class to pass data(images/vector) in batches

class DataSequence(Sequence):
    def __init__(self, imagefiles, labels, batch_size):
        self.bsz = batch_size  # batch size

        # Take labels and a list of image locations in memory
        self.labels = labels
        self.im_list = imagefiles

    def __len__(self):
        # compute number of batches to yield
        return int(math.ceil(len(self.im_list) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([img_to_array(load_img(im, target_size=(110, 110))) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y

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
    model.add(Dense(505, activation='sigmoid'))
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
X_train_files = [train_path + '/' + filename for filename in train_generator.filenames]


val_generator.reset()
y_val = val_generator.labels
X_val_files = [val_path + '/' + filename for filename in val_generator.filenames]

test_generator.reset()
y_test = test_generator.labels
X_test_files = [test_path + '/' + filename for filename in test_generator.filenames]

# num_of_classes = len(train_generator.class_indices)
test_transcripts = [get_generator_value(test_generator.class_indices, int(i)) for i in y_test]
test_transcripts = np.array(test_transcripts)
y_train = [phoc_generate_label(get_generator_value(train_generator.class_indices, int(i))) for i in y_train]
y_val = [phoc_generate_label(get_generator_value(val_generator.class_indices, int(i))) for i in y_val]
y_test = [phoc_generate_label(get_generator_value(test_generator.class_indices, int(i))) for i in y_test]

weight_path = 'phoc_weights.pkl'
if os.path.exists(weight_path):
    model = base_model(70, 90, weight_path= weight_path)
else:
    model = base_model(70, 90)
batch_size = 32

train_sequence = DataSequence(imagefiles=X_train_files, labels= y_train,  batch_size=batch_size)
valid_sequence = DataSequence(imagefiles=X_val_files, labels= y_val, batch_size=batch_size)

for i in range(5):
    model.fit(
        train_sequence,
        batch_size=batch_size,
        epochs=5,
        shuffle= True,
        validation_data=valid_sequence,
        verbose=1)

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