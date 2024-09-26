import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

if __name__ == '__main__':

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Define different regularization strengths
    regularization_strength = 0.01
    
    # Define different CNN architectures with varying stride and filter size
    models_to_evaluate = [
        ##### normal use
        {'name': 'CNN_MODEL', 'layers': [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                        layers.MaxPooling2D((2, 2)),
                                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                                        layers.MaxPooling2D((2, 2), strides=2),
                                        layers.Flatten(),
                                        # layers.Dense(64, activation='relu'),
                                        # layers.Dense(10, activation='softmax'),
                                        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength)),
                                        layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
                                        ]},
        ##### test stride size change
        # {'name': 'CNN_MODEL', 'layers': [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu'),
        #                                 layers.MaxPooling2D((2, 2)),
        #                                 layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
        #                                 layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
        #                                 layers.MaxPooling2D((2, 2)),
        #                                 layers.Flatten(),
        #                                 layers.Dense(64, activation='relu'),
        #                                 layers.Dense(10, activation='softmax'),
        #                                 # layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength)),
        #                                 # layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
        #                                 ]},
        ##### test filter size change
        # {'name': 'CNN_MODEL', 'layers': [layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (5, 5), activation='relu'),
        #                                 layers.MaxPooling2D((2, 2)),
        #                                 layers.Conv2D(32, (5, 5), activation='relu'),
        #                                 layers.Conv2D(32, (5, 5), activation='relu'),
        #                                 layers.MaxPooling2D((2, 2)),
        #                                 layers.Flatten(),
        #                                 layers.Dense(64, activation='relu'),
        #                                 layers.Dense(10, activation='softmax'),
        #                                 # layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength)),
        #                                 # layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
        #                                 ]},
        ##### test feature map change
        # {'name': 'CNN_MODEL', 'layers': [layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #                                 layers.Flatten(),
        #                                 layers.Dense(64, activation='relu'),
        #                                 layers.Dense(10, activation='softmax'),
        #                                 # layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength)),
        #                                 # layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
        #                                 ]},
    ]

    # Training and evaluation loop for different models
    results = []

    for model_data in models_to_evaluate:
        model = models.Sequential(model_data['layers'])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        history = model.fit(train_images, train_labels, epochs=5,
                            validation_data=(test_images, test_labels))

        results.append({'name': model_data['name'],
                        'history': history,
                        'model': model})
        
        model.save('best_weights_mnist.h5')

    # Plot learning curves and accuracy
    for result in results:
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(result['history'].history['loss'], label='Cross Entrophy')
        # plt.plot(result['history'].history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curve')

        plt.subplot(1, 2, 2)
        plt.plot(result['history'].history['accuracy'], label='Training Accuracy')
        plt.plot(result['history'].history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy Rate')
        plt.legend()
        plt.title('Training Accuracy')

    # Plot weight and bias distributions
    for result in results:
        for layer in result['model'].layers:
            if len(layer.get_weights()) > 0:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.hist(weights.ravel(), bins=30)
                plt.xlabel('Value')
                plt.ylabel('Number')
                plt.title(f'Histogram of {layer.name}')

                plt.subplot(1, 2, 2)
                plt.hist(biases.ravel(), bins=30)
                plt.xlabel('Value')
                plt.ylabel('Number')
                plt.title(f'Histogram of {layer.name}')
                
    plt.show()