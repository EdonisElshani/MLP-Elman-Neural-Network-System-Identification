# Multi-Layer Perceptron (MLP) & Elman Neural Networks for System Identification

This project implements **Multi-Layer Perceptron (MLP)** and **Elman Neural Networks** in MATLAB for basic system identification tasks.  
All models are built in MATLAB showcasing manual implementation of **forward propagation**, **backpropagation**, and **weight updates**.


## MLP 1-3-1

- **1 input neuron**
- **1 hidden layer** with configurable size (here: 3)
- **1 output neuron**
- **Tanh activation function** for the hidden layer
- **Linear activation** for the output layer

<p align="center">
  <img src="Illustration/MLP131.png" alt="MLP Architecture" width="500"/>
</p>

---

## How It Works

1. **Initialization**  
   Random weights and biases are assigned for each layer.

2. **Forward Pass**  
   - Hidden layer activations computed with **tanh**.  
   - Output computed with a linear transformation.

3. **Loss Function**  
   - Mean Squared Error (MSE) between predictions and target outputs.

4. **Backpropagation**  
   - Gradients for all weights and biases are computed manually.  
   - Parameters are updated using **gradient descent**.

5. **Training**  
   - Runs for a defined number of epochs.  
   - Prints loss at regular intervals.

---

## Parameters in `Model.m`

| Parameter        | Description                                  |
|------------------|----------------------------------------------|
| `inputSize`      | Number of input neurons                      |
| `hiddenSize`     | Number of neurons in the hidden layer         |
| `outputSize`     | Number of output neurons                      |
| `learningRate`   | Step size for gradient descent                |
| `epochs`         | Number of training iterations                 |

---

### Simulink Model — `KNN.slx`

This repository contains a basic Simulink model **`KNN.slx`**, which implements a **Multi-Layer Perceptron (MLP)** for system identification.

<p align="center">
  <img src="Illustration/Simulink.png" alt="MLP Architecture" width="500"/>
</p>

- **Data Input (`data`)**: Supplies the input signals for both the real system and the MLP model.
- **System Block**: Represents the real system to be identified.  
  - Input: `p` (control signal)  
  - Output: `y` (system response)
- **Model Block (MLP)**: Neural network implementation that predicts the system output.  
  - Inputs: `y` (system output) and `p` (control signal)  
  - Output: `y_hat` (predicted system response)
- **Comparison**: The predicted output `y_hat` is compared against the true output `y` to evaluate the identification accuracy.

This configuration allows the MLP to be trained and validated directly within Simulink, enabling seamless integration of neural network modeling with simulation workflows.

## Usage

MATLAB R2024b

1. Clone this repository:
   ```bash
   git clone https://github.com/<USER>/MLP-Elman-Neural-Network-System-Identification.git
   ```

