# 🚀 Dynamic Neural Engine (N-Layer MLP from Scratch)

A fully dynamic, arbitrary-depth Multi-Layer Perceptron (MLP) built entirely from first principles using **NumPy**. This engine implements dynamic layer configuration and optimizes convergence using custom **Nesterov Accelerated Gradient (NAG)** step logic.

---

## 🧠 Architecture & Core Mechanics

The entire engine operates without high-level deep learning frameworks. By managing structural parameters through tracked list states, the model supports an arbitrary number of hidden layers.

### 1. Look-Ahead Optimization (NAG)
Unlike standard momentum which computes gradients at the current position, this engine updates weights to a provisional look-ahead state (`wf_lst` and `bf_lst`) before computing the forward pass.

$$\theta_{\text{look-ahead}} = \theta_t - \gamma v_t$$

The forward and backward propagation chains run entirely on these look-ahead matrices, preventing gradient overshooting at steep local valleys.

### 2. State-Tracking Lists
* `self.w_lst` / `self.b_lst`: Master weight and bias matrices.
* `self.wf_lst` / `self.bf_lst`: Look-ahead weight and bias matrices used for execution loops.
* `self.v_lst` / `self.bv_lst`: Running velocity matrices tracking directional momentum vectors.
* `self.h_lst` / `self.z_lst`: Intermediate pre-activation ($h$) and post-activation ($z$) layer outputs stored during forward passes.

---

## 🛠️ Activation & Mathematical Foundations

The model utilizes a non-linear combination of **Leaky ReLU** for hidden layer feature activation and **Sigmoid** for binary output boundary predictions.

| Function | Equation | Python Implementation | Derivative Code |
| :--- | :--- | :--- | :--- |
| **Leaky ReLU** | $f(x) = \max(0.01x, x)$ | `np.where(x > 0, x, 0.01 * x)` | `np.where(x > 0, 1, 0.01)` |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | `1 / (1 + np.exp(-x))` | `x * (1 - x)` |

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure you have Python installed along with the core numerical computation library:
```bash
pip install numpy
```
### 2. installing on your machine
Open your terminal and write the below lines:
```bash
git clone https://github.com/DanielDavid-H/dynamic-neural-engine.git
cd dynamic-neural-engine
```
### 3. Running the program
To run the program, write this on the terminal 
```bash
python dynamic-neural-engine.py
```
1. First the program will ask number of neurons in each layer
2. Then it will ask number of hidden layers for the network
3. Finally the model will show the loss values for training a full adder within 10000 iterations.



