# MNIST Handwritten Digit Recognizer

This project implements a neural network for recognizing handwritten digits from the MNIST dataset using NumPy. The model is trained on training data and can make predictions on test images.

## Description

The project includes the following components:

1. **Data Loading:** The MNIST dataset is loaded from the `train.csv` file.
2. **Data Preprocessing:** Data is converted to a NumPy array, normalized, and split into training and development sets.
3. **Parameter Initialization:** Initial weights and biases for the neural network are created.
4. **Backpropagation:** Gradients are computed, and network parameters are updated.
5. **Model Training:** The model is trained on the training data using gradient descent.
6. **Performance Evaluation:** Model accuracy is computed, and results are visualized.

## Dependencies

To run this project, you need Python and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`

Install the dependencies with:

```bash
pip install numpy pandas matplotlib
```

## Running the Project

1. **Download and Prepare Data:**

   Ensure that the `train.csv` file is in the same directory as the script. This file should contain the MNIST data.

2. **Run the Script:**

   Execute the script from the command line:

   ```bash
   python main.py
   ```

3. **Check Results:**

   The script will print the model's accuracy on the training data and display images with predictions for several examples.

## Functions

- `init_params()`: Initializes the parameters (weights and biases) of the neural network.
- `ReLU(Z)`: Applies the ReLU activation function.
- `softmax(Z)`: Applies the softmax activation function.
- `forward_prop(W1, b1, W2, b2, X)`: Performs forward propagation through the neural network.
- `ReLU_deriv(Z)`: Computes the derivative of the ReLU function.
- `one_hot(Y)`: Converts labels to one-hot encoding format.
- `backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)`: Performs backpropagation and computes gradients.
- `update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)`: Updates the neural network parameters.
- `get_predictions(A2)`: Returns model predictions based on activations.
- `get_accuracy(predictions, Y)`: Computes model accuracy.
- `gradient_descent(X, Y, alpha, iterations)`: Trains the model using gradient descent.
- `make_predictions(X, W1, b1, W2, b2)`: Generates predictions for new data.
- `test_prediction(index, W1, b1, W2, b2)`: Tests the model on a specific image and displays it.

## Examples

At the end of the script, predictions and visualizations are performed for the first four images from the training dataset.

