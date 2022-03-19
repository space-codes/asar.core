import os
import itertools
import numpy as np
from numpy import linalg as LA
import pandas as pd
from tensorflow_addons.layers import SpatialPyramidPooling2D
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import matplotlib.pyplot as plt
from phoc_label_generator import phoc_generate_label

# Uncomment the following line and set appropriate GPU if you want to set up/assign a GPU device to run this code
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Input: Two vectors x and y
# Output: Similarity index = Cosine simialirity * 1000

def similarity(x, y):
    return 1000 * np.dot(x, y) / (LA.norm(x) * LA.norm(y))


# Input: Confusion matrix, true and predicted class names, plot title, color map and normalization parameter(bool)
# Output: Plots and saves confusion matrix

def plot_confusion_matrix(cm, target_names_true, target_names_pred, title='Confusion matrix', cmap=None,
                          normalize=True):
    # accuracy = np.trace(cm) / float(np.sum(cm))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100
    cm[np.isnan(cm)] = 0
    plt.figure(figsize=(12, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks_true = np.arange(len(target_names_true))
    tick_marks_pred = np.arange(len(target_names_pred))
    plt.xticks(tick_marks_pred, target_names_pred)
    plt.yticks(tick_marks_true, target_names_true)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Word Class Label Length')
    plt.savefig("Test_Plots/" + title + ".png")
    plt.xlabel('Predicted Word Class Label Length')
    plt.show()


# Input: Model, dataframes for test set samples, dictionary for test set words and label(PHOC vector)
# Output: Similarity index = Cosine simialirity * 1000

def accuracy_test(model, X_test, transcripts, all_transcripts, name):
    cnt = 0
    no_of_images = len(X_test)
    acc_by_len = dict()
    word_count_by_len = dict()
    for k in transcripts:
        acc_by_len[len(k)] = 0
        word_count_by_len[len(k)] = 0
    lengths_true = sorted(acc_by_len.keys())
    lengths_pred = list(set(len(x) for x in transcripts))
    l = len(lengths_true)
    m = len(lengths_pred)
    idx_true = dict()
    idx_pred = dict()
    for k in range(l):
        idx_true[lengths_true[k]] = k
    for k in range(m):
        idx_pred[lengths_pred[k]] = k
    conf_matrix = np.zeros(shape=(l, m))
    Predictions = []

    # Finding predictions for test set word images

    for (img, transcript) in zip(X_test, transcripts):
        x = img_to_array(load_img(img, target_size=(128, 128)))
        word = transcript
        word_count_by_len[len(word)] += 1
        x = np.expand_dims(x, axis=0)
        print('predicting ' + transcript + '....')
        y_pred = np.squeeze(model.predict(x))
        mx = 0
        for k in all_transcripts:
            temp = similarity(y_pred, phoc_generate_label(k))
            if temp > mx:
                mx = temp
                op = k
        conf_matrix[idx_true[len(word)]][idx_pred[len(op)]] += 1
        Predictions.append((img, word, op))
        if op == word:
            cnt += 1
            acc_by_len[len(word)] += 1
    for k in acc_by_len:
        if acc_by_len[k] != 0:
            acc_by_len[k] = acc_by_len[k] / word_count_by_len[k] * 100

    # Storing true and predicted labels for each image sample

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])
    df.set_index('Image', inplace=True)
    df.to_csv("Test_Results/" + name + ".csv")
    print("Correct predictions:", cnt, "   Accuracy=", cnt / no_of_images)

    # Plotting length-wise correct predictions

    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*acc_by_len.items()))
    plt.title('Acc:' + str(cnt) + '/' + str(no_of_images) + '  Correct predictions lengthwise')
    plt.xticks(lengths_true)
    plt.xlabel('Word Length')
    plt.ylabel('Percentage of correct predictions')
    plt.savefig("Test_Plots/" + name + "_ZSL_acc.png")
    plt.show()

    # Plotting length-wise confusion matrix
    plot_confusion_matrix(conf_matrix, lengths_true, lengths_pred, title=name + "_confmat")
    return cnt / no_of_images


# Input: model, folder names for samples, CSV files having sample to label mapping(Train and test set), and name(identifier in plot names)
# Output: Prediction accuracy (Also calls functions for plotting)

def get_generator_value(class_indicates, index):
    key_list = list(class_indicates.keys())
    val_list = list(class_indicates.values())
    return key_list[val_list.index(index)]

test_path = 'testset/test'
train_path = 'testset/train'
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        test_path,
        shuffle= False,
        target_size=(16, 16),
        batch_size=1,
        class_mode='binary')

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        train_path,
        shuffle= False,
        target_size=(16, 16),
        batch_size=1,
        class_mode='binary')

test_generator.reset()
y_test = test_generator.labels
test_transcripts = [get_generator_value(test_generator.class_indices, int(i)) for i in y_test]
X_test_files = [test_path + '/' + filename for filename in test_generator.filenames]
MODEL = 'phoc-model'
all_transcripts = list(train_generator.class_indices.keys())
name = MODEL + "_1"

# Create directories for storing test results and plots

if not os.path.exists("Test_Plots"):
    os.makedirs("Test_Plots")
if not os.path.exists("Test_Results"):
    os.makedirs("Test_Results")

# Load model from filename and print model name(if successfully loaded)
model = tf.keras.models.load_model(MODEL + ".h5")
print(MODEL)

# Function called for test set prediction and result plotting
accuracy = accuracy_test(model, X_test_files, test_transcripts, all_transcripts, name + "_conv")
print("Conventional Accuracy = ", accuracy)