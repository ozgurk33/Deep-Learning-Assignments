# YZM304 Deep Learning Project 1: Banknote Authentication

## 1. Introduction
Banknote authentication is a critical classification problem for financial security. In this project, Multi-Layer Perceptron (MLP) models were developed to distinguish genuine banknotes from forgeries using features derived from wavelet transform images (variance, skewness, curtosis, and entropy).

The primary objective of this work is to demystify the "black box" nature of deep learning by implementing the backpropagation algorithm from scratch using only NumPy. This custom implementation is then compared against industry-standard PyTorch models to validate mathematical accuracy and explore optimization techniques like L2 regularization and Batch Normalization.

## 2. Methods
[cite_start]To ensure reproducibility, all hyperparameters and preprocessing steps are detailed below[cite: 55, 56]:

* **Dataset & Feature Engineering**: The `BankNote_Authentication.csv` dataset was used. [cite_start]A new feature called `var_skew_interact` was created by multiplying variance and skewness values to enhance class separability[cite: 39].
* [cite_start]**Data Preprocessing**: The dataset was split into 60% Training, 20% Validation (Dev), and 20% Test sets[cite: 41]. [cite_start]Data was standardized using `StandardScaler` fitted only on the training set to prevent data leakage[cite: 39].
* **Model Architectures**:
  1. **Custom NumPy MLP**: A functional/step-by-step implementation with 1 hidden layer (6 neurons), using Tanh and Sigmoid activations.
  2. **PyTorch Basic MLP**: An OOP-based model mimicking the Custom MLP. [cite_start]Weights were initialized identically for fair comparison[cite: 51].
  3. [cite_start]**PyTorch Deep MLP**: A deeper architecture with 2 hidden layers (8 and 4 neurons) to address potential bias/variance issues[cite: 44].
  4. [cite_start]**PyTorch Mini-Batch & BatchNorm MLP**: A model utilizing `BatchNorm1d` and `DataLoader` for accelerated and stable training[cite: 54].
* **Hyperparameters**:
  * [cite_start]**Loss Function**: Binary Cross Entropy (BCE)[cite: 52].
  * [cite_start]**Optimizer**: Stochastic Gradient Descent (SGD)[cite: 51].
  * **Learning Rate**: 0.05.
  * **Epochs (n_steps)**: Determined via an automated search; [cite_start]300 steps were found optimal for reaching >90% validation accuracy[cite: 48].
  * [cite_start]**Regularization**: L2 penalty (`weight_decay=1e-3`) for the deep model[cite: 43].
  * [cite_start]**Mini-Batch Size**: 64[cite: 54].

## 3. Results
[cite_start]The models were evaluated on the test set (275 samples), achieving high accuracy and perfect recall for the forgery class (Class 1)[cite: 50, 57].

### Custom NumPy (Functional) MLP Results
The following image shows the detailed metrics for the step-by-step NumPy implementation:
![Custom MLP Results](images/np_mlp_result.png)

### OOP (Class-based) and PyTorch Model Comparison
The evaluation of the multiple PyTorch architectures and the comparison with the OOP-based Custom MLP class is provided below:
![PyTorch Results 1](images/class_and_pytorch_results_1.png)
![PyTorch Results 2](images/class_and_pytorch_results_2.png)

### Overfitting and Underfitting Analysis
To monitor the model's learning progress and ensure it didn't memorize the data, training and validation loss curves were plotted:
![Loss Curves](images/overfittin_underfitting.png)

## 4. Discussion
The project successfully demonstrated that the custom-coded backpropagation algorithm yields identical results to PyTorch when initialized with the same weights. This proves the correctness of the partial derivative calculations and matrix operations.

The inclusion of `var_skew_interact` significantly improved classification speed, allowing even simple models to reach near-perfect recall. [cite_start]The training curves indicate a healthy learning process where validation loss consistently follows the training loss, confirming that the L2 regularization effectively prevented overfitting[cite: 43]. Future work could involve testing momentum-based optimizers like Adam or alternative activation functions like ReLU to compare convergence speeds.
