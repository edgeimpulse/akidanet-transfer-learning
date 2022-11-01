import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Activation, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam

from keras import Model
from akida_models.layer_blocks import dense_block

from akida_models import akidanet_imagenet
from keras.utils.data_utils import get_file

# Create a quantized base model without top layers
base_model = akidanet_imagenet(input_shape=MODEL_INPUT_SHAPE,
                               classes=classes,
                               alpha=0.5,
                               include_top=False,
                               pooling='avg',
                               weight_quantization=4,
                               activ_quantization=4,
                               input_weight_quantization=8)

# Get pretrained quantized weights and load them into the base model
pretrained_weights = get_file(
    "akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5",
    "http://data.brainchip.com/models/akidanet/akidanet_imagenet_224_alpha_50_iq8_wq4_aq4.h5",
    cache_subdir='models')

base_model.load_weights(pretrained_weights, by_name=True)
base_model.trainable = False

from keras import Model
from keras.layers import Activation, Dropout, Reshape, Flatten
from akida_models.layer_blocks import dense_block

x = base_model.output
x = Flatten(name='flatten')(x)
x = dense_block(x,
                units=16,
                name='fc1',
                add_batchnorm=False,
                add_activation=True)
x = Dropout(0.5, name='dropout_1')(x)
x = dense_block(x,
                units=classes,
                name='predictions',
                add_batchnorm=False,
                add_activation=False)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape')(x)

# Build the model
model = Model(base_model.input, x, name='')

lr = tf.keras.optimizers.schedules.PolynomialDecay(
    learning_rate,
    epochs,
    end_learning_rate=learning_rate / 10.0,
    power=1.0,
    cycle=False,
    name=None,
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# train the neural network
model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2, callbacks=callbacks)

print('')
print('Initial training done.', flush=True)
print('')