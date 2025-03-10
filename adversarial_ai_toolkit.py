import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

class AdversarialAI:
    def __init__(self):
        self.model = self.build_model()
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
        return x_train, y_train, x_test, y_test

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=5, batch_size=32)

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

    def generate_adversarial_examples(self, images, labels, epsilon=0.1):
        perturbations = np.random.normal(0, epsilon, images.shape)
        adversarial_images = images + perturbations
        adversarial_images = np.clip(adversarial_images, 0, 1)
        return adversarial_images

    def plot_examples(self, images, labels, title):
        plt.figure(figsize=(10, 5))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].reshape(28, 28), cmap='gray')
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
        plt.suptitle(title)
        plt.show()

    def run(self):
        self.train_model()
        self.evaluate_model()
        adversarial_examples = self.generate_adversarial_examples(self.x_test[:10], self.y_test[:10])
        self.plot_examples(adversarial_examples, self.y_test[:10], 'Adversarial Examples')

if __name__ == '__main__':
    adversarial_ai = AdversarialAI()
    adversarial_ai.run()