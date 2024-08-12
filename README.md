# Flower Classification with TensorFlow and ResNet50

## Overview

This project demonstrates how to build a flower classification model using TensorFlow and a pre-trained ResNet50 model. The model classifies images into five categories: daisies, dandelions, roses, sunflowers, and tulips.

## Requirements

To run this notebook, you will need:

- TensorFlow
- Matplotlib
- NumPy
- OpenCV
- PIL
- pathlib

You can install the required packages using pip:

```bash
pip install tensorflow matplotlib numpy opencv-python pillow
```

## Getting Started

1. **Download and Extract the Dataset:**

   The dataset used is the "flower_photos" dataset from TensorFlow. The dataset is automatically downloaded and extracted.

   ```python
   dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
   data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
   data_dir = pathlib.Path(data_dir)
   ```

2. **Prepare the Data:**

   The data is split into training and validation sets. Images are resized and batched.

   ```python
   img_height, img_width = 180, 180
   batch_size = 32

   train_ds = tf.keras.preprocessing.image_dataset_from_directory(
     data_dir,
     validation_split=0.2,
     subset="training",
     seed=123,
     image_size=(img_height, img_width),
     batch_size=batch_size,
     label_mode='categorical')

   val_ds = tf.keras.preprocessing.image_dataset_from_directory(
     data_dir,
     validation_split=0.2,
     subset="validation",
     seed=123,
     image_size=(img_height, img_width),
     batch_size=batch_size,
     label_mode='categorical')
   ```

3. **Build and Compile the Model:**

   The model uses a pre-trained ResNet50 network as a base. The model is compiled with an Adam optimizer and categorical crossentropy loss.

   ```python
   resnet_model = Sequential()
   pretrained_model = tf.keras.applications.ResNet50(
       include_top=False,
       input_shape=(img_height, img_width, 3),
       pooling='avg',
       weights='imagenet'
   )
   for layer in pretrained_model.layers[-2:]:
       layer.trainable = True

   resnet_model.add(pretrained_model)
   resnet_model.add(Flatten())
   resnet_model.add(Dense(512, activation='relu'))
   resnet_model.add(Dense(5, activation='softmax'))

   resnet_model.compile(optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
   ```

4. **Train the Model:**

   Train the model on the training dataset and validate on the validation dataset.

   ```python
   epochs = 10
   history = resnet_model.fit(
     train_ds,
     validation_data=val_ds,
     epochs=epochs
   )
   ```

5. **Evaluate and Plot Results:**

   Plot the training and validation accuracy and loss over epochs.

   ```python
   plt.figure(figsize=(12, 6))
   plt.subplot(1, 2, 1)
   plt.plot(history.history['accuracy'])
   plt.plot(history.history['val_accuracy'])
   plt.title('Model Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend(['Train', 'Validation'])
   plt.grid(True)

   plt.subplot(1, 2, 2)
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.title('Model Loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend(['Train', 'Validation'])
   plt.grid(True)
   plt.show()
   ```

6. **Make Predictions:**

   Load sample images and make predictions using the trained model.

   ```python
   # Load and preprocess images
   image = cv2.imread('/path/to/your/image.jpg')
   image_resized = cv2.resize(image, (img_height, img_width))
   image_array = np.expand_dims(image_resized, axis=0)

   # Predict
   prediction = resnet_model.predict(image_array)
   class_names = train_ds.class_names
   output_class = class_names[np.argmax(prediction)]
   print("The predicted class is", output_class)
   ```

## Jupyter Notebooks Support

### PyCharm Community Edition

PyCharm Community Edition supports Jupyter notebooks in read-only mode. To get full support for local notebooks, download and try [PyCharm Professional](https://www.jetbrains.com/pycharm/buy/).

### DataSpell

Try [DataSpell](https://www.jetbrains.com/dataspell/), a dedicated IDE for data science, with full support for local and remote notebooks.

### Datalore

Try [Datalore](https://www.jetbrains.com/datalore/), an online environment for Jupyter notebooks in the browser.
