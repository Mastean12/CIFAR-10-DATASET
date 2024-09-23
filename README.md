### README: CIFAR-10 Classification with Keras

---

This project performs image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The procedure follows a similar approach to classifying the MNIST dataset, but modifications are made due to the different dataset shape and color channels.

#### Files included:
- **cifar10_classification.py**: Python script that downloads CIFAR-10, builds the CNN model, trains the model, and evaluates the accuracy.
- **model_summary.txt**: Output text file containing the model summary.
- **requirements.txt**: Lists the Python packages required for this project.

#### Steps Performed:
1. **Dataset Loading**:
   The CIFAR-10 dataset is loaded using Keras' built-in dataset loader:
   ```python
   from keras.datasets import cifar10
   ```
   The data is then split into training and testing sets:
   ```python
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   ```

2. **Data Preprocessing**:
   - Normalize the pixel values to a range between 0 and 1 by dividing by 255.
   - Convert the class vectors to binary class matrices (one-hot encoding) using `to_categorical`:
     ```python
     from keras.utils import to_categorical
     y_train = to_categorical(y_train, 10)
     y_test = to_categorical(y_test, 10)
     ```

3. **Model Architecture**:
   The CNN model is constructed as follows:
   - **Input Shape**: `(32, 32, 3)` to handle the 32x32 RGB images.
   - **Convolutional Layers**: Two sets of `Conv2D` layers with ReLU activation followed by max-pooling.
   - **Flatten**: The output from the convolutional layers is flattened.
   - **Fully Connected Layers**: Two dense layers, with the final layer having 10 neurons (one for each class) and softmax activation.
   
   ```python
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```

4. **Compilation**:
   The model is compiled with categorical crossentropy loss, Adam optimizer, and accuracy as the metric:
   ```python
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   ```

5. **Training**:
   The model is trained with a batch size of 32 and for 10 epochs by default:
   ```python
   model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
   ```

6. **Evaluation**:
   After training, the model is evaluated on the test set to obtain the final test accuracy:
   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print('Test accuracy:', test_acc)
   ```

#### Results:

- **Training Accuracy**: Achieved after 10 epochs.
- **Testing Accuracy**: Achieved after 10 epochs.

(Optional) You can experiment with more epochs by changing the `epochs` parameter in the `model.fit()` method to see how it affects the accuracy.

---

### How to Run the Code:
1. **Install dependencies**:
   Run the following command to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script**:
   Execute the following command to start the training and evaluation:
   ```bash
   python cifar10_classification.py
   ```

3. **Optional**: Change the number of epochs or modify other hyperparameters in the script as needed.

---

### Requirements:
- Python 3.x
- Keras
- TensorFlow
- NumPy

For additional details on the model's performance, refer to the generated `model_summary.txt`.

