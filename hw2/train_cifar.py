import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 25:
        lr *= 0.5
    elif epoch > 50:
        lr *= 0.5
    elif epoch > 75:
        lr *= 0.5
    return lr

if __name__ == '__main__':
    
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Define different regularization strengths
    regularization_strength = 0.01

    ############### preprocessing ################
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # decrease light
    mean = np.mean(train_images, axis=(0, 1, 2))
    train_images -= mean
    test_images -= mean
    
    # dynamic learning change
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # image augmented
    datagen = ImageDataGenerator(
        rotation_range=90,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        fill_mode='nearest'
    )
    datagen.fit(train_images)
    train_generator = datagen.flow(train_images, train_labels, batch_size=32)
    ############### preprocessing end ################

    # Define different CNN architectures with varying stride and filter size
    models_to_evaluate = [
        {'name': 'CNN_MODEL', 'layers': [layers.Conv2D(128, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3)),
                                        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
                                        layers.MaxPooling2D((2, 2), strides=2, padding='valid'),
                                        layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
                                        layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
                                        layers.MaxPooling2D((2, 2), strides=2, padding='valid'),
                                        layers.Flatten(),
                                        # layers.Dense(64, activation='relu'),
                                        # layers.Dense(10, activation='softmax'),
                                        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength)),
                                        layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(regularization_strength))
                                        ]},
    ]

    # Training and evaluation loop for different models
    results = []

    for model_data in models_to_evaluate:
        model = models.Sequential(model_data['layers'])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        # history = model.fit(train_generator, epochs=100,
        #                     validation_data=(test_images, test_labels), callbacks=[lr_scheduler])
        history = model.fit(train_images, train_labels, epochs=100,
                            validation_data=(test_images, test_labels), callbacks=[lr_scheduler])

        results.append({'name': model_data['name'],
                        'history': history,
                        'model': model})
        
        model.save('best_weights_cifar.h5')

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