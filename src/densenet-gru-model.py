from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization, Flatten, GRU, Add, Concatenate, Reshape
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

def densenet121_model(img_rows=224, img_cols=224, channels=3, num_classes=1000, dropout_keep_prob=0.2):
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=(img_rows, img_cols, channels))  # this assumes K.image_data_format() == 'channels_last'
    # create the base pre-trained model
    rnn_size = 64
    base_model = DenseNet121(input_tensor=input_tensor,weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_keep_prob)(x)
    # x = Flatten()(x)
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    # x = Dense(1024, activation='relu')(x)
    '''
    '''
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1a = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_a')(x)
    gru1_merged = Add()([gru_1, gru_1a])

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2a = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_a')(
        gru1_merged)
    x = Concatenate(axis=1, name='DYNN/output')([gru_2, gru_2a])
    # x=Dropout(0.5)(x)

    gru_3 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru3')(x)
    gru_3b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_3b')(x)
    gru3_merged = Add()([gru_3, gru_3b])

    gru_4 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru4')(gru3_merged)
    gru_4b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru4_b')(
        gru3_merged)
    x = Concatenate(axis=1, name='DYNN2/output')([gru_4, gru_4b])
    x = Dropout(dropout_keep_prob)(x)
    x = Flatten()(x)
    x = Dense(units=num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x, name='DenseNet121')
    from tensorflow.keras.optimizers import SGD, Adam, Adadelta

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    from keras.utils.vis_utils import plot_model as plot
    from IPython.display import Image

    plot(model, to_file="mdelqqGRU.png", show_shapes=True)
    Image('mdeqql.png')
    return model

train_datagen = ImageDataGenerator(
      #rescale=1./255,
      preprocessing_function=preprocess_input,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

#val_datagen = ImageDataGenerator(rescale=1. / 255.)
val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        'dataset/train',
        # All images will be resized to 150x150
        target_size=(128, 128),
        batch_size=32,
        # binary: use binary_crossentropy loss, we need binary labels
        # categorical : use categorical_crossentropy loss, then need categorical labels
        class_mode='categorical')

val_generator = train_datagen.flow_from_directory(
        # This is the target directory
        'dataset/val',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

test_generator = train_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

num_of_classes = len(train_generator.class_indices)

model = densenet121_model(img_rows=310, img_cols=310, channels= 3, num_classes=num_of_classes, dropout_keep_prob=0.5)

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=val_generator,
      validation_steps=50,
      verbose=1)

model.save('arabic-manuscripts-3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

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