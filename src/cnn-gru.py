from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Input, BatchNormalization, MaxPooling2D, Reshape, GRU, Concatenate, Add, Flatten
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

def base_model(img_width, img_height,weight_path=None):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    rnn_size = 64
    rnn_size1 = 128
    # model = Sequential()
    input_shapes = (img_height, img_width, 3)
    input = Input(shape=input_shapes)

    x = Convolution2D(64, (3, 3), strides=(1, 1), dilation_rate=(1, 1), activation='relu', use_bias=True,
                      kernel_regularizer=l2(0.0002), padding='same')(input)
    x = Convolution2D(256, (3, 3), strides=(1, 1), dilation_rate=(1, 1), activation='relu', use_bias=True,
                      kernel_regularizer=l2(0.0002), padding='same')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid', )(x)
    x = Dropout(0.25)(x)
    # #x=Flatten()(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

    # x = Dense(1024, activation='relu')(x)
    '''
    '''
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1a = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_a')(x)
    gru1_merged = Add()([gru_1, gru_1a])

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2a = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_a')(gru1_merged)
    x = Concatenate(axis=1, name='DYNN/output')([gru_2, gru_2a])
    # x=Dropout(0.5)(x)


    gru_3 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru3')(x)
    gru_3b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_3b')(x)
    gru3_merged = Add()([gru_3, gru_3b])

    gru_4 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru4')(gru3_merged)
    gru_4b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru4_b')(gru3_merged)
    x = Concatenate(axis=1, name='DYNN2/output')([gru_4, gru_4b])

    x = Dropout(0.5)(x)

    # x = Dense(937, init='he_normal', activation='softmax')(x)
    x = Flatten()(x)
    x = Dense(932, activation='softmax')(x)
    # loss1_classifier_act = Activation('softmax')(loss1_classifier)

    model = Model(inputs=input, outputs=x)
    # model = create_googlenet()
    from tensorflow.keras.optimizers import SGD, Adam, Adadelta

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    if weight_path:
        model = load_model(weight_path)

    model.summary()
    from keras.utils.vis_utils import plot_model as plot
    from IPython.display import Image

    plot(model, to_file="mdelqqGRU.png", show_shapes=True)
    Image('mdeqql.png')
    return model

train_datagen = ImageDataGenerator(
      #rescale=1./255,
      #preprocessing_function=preprocess_input,
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
        target_size=(310, 310),
        batch_size=32,
        # binary: use binary_crossentropy loss, we need binary labels
        # categorical : use categorical_crossentropy loss, then need categorical labels
        class_mode='categorical')

val_generator = train_datagen.flow_from_directory(
        # This is the target directory
        'dataset/val',
        target_size=(310, 310),
        batch_size=32,
        class_mode='categorical')

test_generator = train_datagen.flow_from_directory(
        'dataset/test',
        target_size=(310, 310),
        batch_size=32,
        class_mode='categorical')

model = base_model(310,310)

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=val_generator,
      validation_steps=50,
      verbose=1)


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