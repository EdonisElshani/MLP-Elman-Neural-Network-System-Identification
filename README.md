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
  <img src="MLP131.png" alt="MLP Architecture" width="450"/>
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

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
