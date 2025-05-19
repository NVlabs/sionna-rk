#  Overview
 
## Description:
The Neural Demapper is a model that computes log-likelihood ratios (LLRs) for bits in received quadrature amplitude modulated (QAM) symbols, which are commonly used in modern wireless communication systems.
 
This model is ready for commercial/non-commercial use.
 
### License/Terms of Use:
The software available under this repository is governed by the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
 
In connection with your use of this software, you may receive links to third party technology, and your use of third-party technology may be subject to third-party terms, privacy statements or practices. NVIDIA is not responsible for the terms, privacy statements or practices of third parties. You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale products or services that include or incorporate any third-party technology. NVIDIA does not grant to you under the project license any necessary patent or other rights, including standard essential patent rights, with respect to any such third-party technology.
 
You are responsible for ensuring that your use of this model complies with all applicable laws. You are responsible for ensuring safe integration and thorough testing prior to deployment.
 
### Deployment Geography:
Global
 
### Use Case:
Can be used as drop-in replacement for a classical demapper in wireless communications. This neural network is mostly for educational purposes.
 
### Release Date:
04/30/2025
 
## Model Architecture:
**Architecture Type:** Sequential multi-layer perceptron (MLP)
 
**Network Architecture:** MLP
 
## Input:
**Input Type(s):** Real part of received QAM symbol, imaginary part of received QAM symbol
 
**Input Format(s):** Floating point, Floating point
 
**Input Parameters:** 1D, 1D
 
**Other Properties Related to Input:** The demapper takes as input the 2 inputs (real and imaginary part of the received noisy QAM symbol as occurring in wireless communication systems).
 
## Output:
**Output Type(s):** Log-likelihood ratios (LLRs) per bit of the QAM constellation.
 
**Output Format(s):** Floating point
 
**Output Parameters:** 1D
 
## Software Integration:
**Runtime Engine(s):**
Tensorflow or TensorRT
 
**Supported Hardware Microarchitecture Compatibility:**
* [NVIDIA Ampere] <br>
* [NVIDIA Blackwell] <br>
* [NVIDIA Jetson] <br>
* [NVIDIA Hopper] <br>
 
**[Preferred/Supported] Operating System(s):**
* [Linux] <br>
* [Linux 4 Tegra] <br>
* [QNX] <br>
* [Windows] <br>
 
## Model Version(s):
Neural Demapper v1.0
 
# Training and Evaluation Datasets:
 
## Training Dataset:
 
**Data Collection Method by dataset**
Synthetic
 
**Labeling Method by dataset**
Synthetic
**Properties:** Synthetic QAM symbols generated from a predefined set of possible values, with additive Gaussian white noise (AWGN) applied dynamically during training to simulate real-world wireless channel conditions. Each sample represents a noisy transmission of a QAM-modulated signal, where the AWGN models the stochastic effects of the wireless environment. This process ensures that the demapper is trained on a diverse range of signal variations, improving robustness to real-world impairments in modern communication systems.
## Evaluation Dataset:
**Data Collection Method by dataset**
Synthetic
 
**Labeling Method by dataset**
Synthetic
 
**Properties:** Synthetic QAM symbols generated from a predefined set of possible values, with additive Gaussian white noise (AWGN) applied dynamically during training to simulate real-world wireless channel conditions. Each sample represents a noisy transmission of a QAM-modulated signal, where the AWGN models the stochastic effects of the wireless environment. This process ensures that the demapper is trained on a diverse range of signal variations, improving robustness to real-world impairments in modern communication systems.
 
## Inference:
**Engine:** Tensor(RT) or Tensorflow
 
**Test Hardware:**
NVIDIA Jetson AGX Orin
 
## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 
 
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.
 
Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
