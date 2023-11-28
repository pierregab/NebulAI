import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
import cv2
import os

def preprocess_input(img):
    img = cv2.resize(img, (512, 512))
    img = img / 127.5 - 1
    return img

def create_data_generators(base_dir, batch_size):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       zoom_range=0.2)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(os.path.join(base_dir, 'train'),
                                                        target_size=(512, 512),
                                                        batch_size=batch_size)
    validation_generator = val_datagen.flow_from_directory(os.path.join(base_dir, 'test'),
                                                           target_size=(512, 512),
                                                           batch_size=batch_size)
    return train_generator, validation_generator

def build_model(hp=None):
    base_model = tf.keras.applications.VGG16(input_shape=(512, 512, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
    ])

    # Hyperparameters or default values
    dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1) if hp else 0.4
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG') if hp else 0.0005

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(model, train_generator, validation_generator, epochs=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stopping])
    return history

def hyperparameter_tuning(train_generator, validation_generator, max_trials=10, epochs=10):
    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=epochs,
                         factor=3,
                         directory='hyperparam_tuning',
                         project_name='nebula_classification',
                         hyperband_iterations=2)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(train_generator,
                 epochs=epochs,
                 validation_data=validation_generator,
                 callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

def run_training(tune_hyperparams=False, base_dir='image_dataset', batch_size=8):
    train_generator, validation_generator = create_data_generators(base_dir, batch_size)

    if tune_hyperparams:
        best_hps = hyperparameter_tuning(train_generator, validation_generator)
        model = build_model(best_hps)
    else:
        model = build_model()

    history = train_model(model, train_generator, validation_generator, epochs=50)
    model.save("nebula_model.h5")
    return history

if __name__ == '__main__':
    history = run_training(tune_hyperparams=True, base_dir='my_image_dataset', batch_size=16)
