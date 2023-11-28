import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

def preprocess_input(img):
    img = cv2.resize(img, (512, 512))
    img = img / 127.5 - 1
    return img

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    return tf.keras.models.load_model(model_path)

def create_test_generator(base_dir, batch_size):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(512, 512),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    return test_generator

def evaluate_model(model, test_generator):
    test_predictions = model.predict(test_generator)
    test_predicted_labels = np.argmax(test_predictions, axis=-1)
    test_true_labels = test_generator.classes
    accuracy = np.mean(test_predicted_labels == test_true_labels)
    return accuracy, test_true_labels, test_predicted_labels

def print_evaluation_report(accuracy, true_labels, predicted_labels, class_labels):
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=class_labels))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

def run_test(base_dir='image_dataset', model_path='nebula_best2.h5', batch_size=8):
    model = load_model(model_path)
    test_generator = create_test_generator(base_dir, batch_size)
    accuracy, true_labels, predicted_labels = evaluate_model(model, test_generator)
    print_evaluation_report(accuracy, true_labels, predicted_labels, test_generator.class_indices.keys())

if __name__ == '__main__':
    run_test(base_dir='image_dataset', model_path='nebula_best2.h5', batch_size=8)
