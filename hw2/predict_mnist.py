import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

if __name__ == '__main__':

    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    # Load your trained model
    model = models.load_model('best_weights_mnist.h5')

    # Make predictions on the test dataset
    predictions = model.predict(test_images)
    
    # check accuracy
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    # Find correctly and misclassified images
    correctly_classified = []
    misclassified = []

    for i in range(len(test_labels)):
        true_label = test_labels[i]
        predicted_label = np.argmax(predictions[i])
        
        if true_label == predicted_label:
            correctly_classified.append((test_images[i], true_label, predicted_label))
        else:
            misclassified.append((test_images[i], true_label, predicted_label))

    # Display some correctly classified and misclassified images
    plt.figure(figsize=(12, 6))
    plt.suptitle('Correctly vs. Misclassified Images')

    # Display correctly classified images
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(correctly_classified[i][0], cmap='gray')
        plt.title(f'True: {correctly_classified[i][1]}\nPredicted: {correctly_classified[i][2]}')
        plt.axis('off')

    # Display misclassified images
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(misclassified[i][0], cmap='gray')
        plt.title(f'True: {misclassified[i][1]}\nPredicted: {misclassified[i][2]}')
        plt.axis('off')

    plt.show()
    
    
    # show feature maps
    img = train_images[0:1]

    layer_names = [layer.name for layer in model.layers if 'conv2d' in layer.name]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img)

    for layer_name, activation in zip(layer_names, activations):
        num_features = activation.shape[-1]
        plt.figure(figsize=(15, 10))

        for feature_index in range(num_features):
            feature_map = activation[0, :, :, feature_index]  # Remove the batch dimension
            plt.subplot(num_features // 8, 8, feature_index + 1)
            plt.imshow(feature_map, cmap='viridis')
            plt.axis('off')

        plt.suptitle('Feature Map')
        plt.show()
