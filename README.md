# 🧠 Deep-Facial-Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8.svg)](https://opencv.org/)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/Fawzy-AI-Explorer/Deep-Facial-Recognition/issues)
[![Stars](https://img.shields.io/github/stars/Fawzy-AI-Explorer/Deep-Facial-Recognition?style=social)](https://github.com/Fawzy-AI-Explorer/Deep-Facial-Recognition/stargazers)

## 📋 Overview
Deep-Facial-Recognition is an advanced machine learning project that implements a Siamese neural network architecture for facial recognition and verification. The system compares facial images to determine if they belong to the same person, with applications ranging from security systems to user authentication.

## ✨ Features
- **🔍 Face Verification**: Compare faces to determine if they belong to the same person
- **🔄 Siamese Neural Network**: Utilizes twin neural networks with shared weights to learn facial embeddings
- **⚡ Real-time Processing**: Integration with webcam for live face verification
- **🖥️ Interactive GUI**: User-friendly interface built with Kivy
- **🗃️ Dataset Management**: Tools for collecting and organizing facial datasets

## 📁 Project Structure
```
Deep-Facial-Recognition/
│
├── APP/                            # GUI Application files
│   ├── app.py                      # Main application entry point
│   ├── models.py                   # Model definitions for the application
│   └── application_data/           # Application-specific data
│       ├── input_image/            # Storage for input images
│       └── verification_images/    # Images used for verification
│
├── application_data/               # Main application data directory
│   ├── input_image/                # Storage for current input image
│   └── verification_images/        # Database of verification images
│
├── data/                           
│   ├── anchor/                     # Reference faces
│   ├── positive/                   # Same person as anchor
│   └── negative/                   # Different people than anchor
│
├── model/                          
│   └── siamese_network.pth         # Saved Siamese network model
│
├── training_checkpoints/           
│   └── ckpt_epoch_*.pth            # Model saved at different epochs
│
├── Deep Facial Recognition.ipynb   # Jupyter notebook with model development
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## 📊 Dataset Structure
The project uses three types of image data:
```
data/  
   ├── positive/  (.jpg files)  
   ├── negative/  (.jpg files)  
   └── anchor/    (.jpg files)
```

- **Anchor images**: Reference faces of the person to be recognized
- **Positive images**: Different images of the same person as in anchor images
- **Negative images**: Images of different people (not matching anchor)

## 🛠️ Technologies Used
- **Python**: Core programming language
- **PyTorch**: Deep learning framework for model development
- **OpenCV**: Computer vision for image capture and processing
- **Kivy**: GUI framework for the application interface
- **Kaggle Datasets**: Uses the LFW (Labeled Faces in the Wild) dataset
- **PIL**: Python Imaging Library for image manipulation
- **NumPy**: For efficient numerical operations
- **Matplotlib**: For data visualization and analysis

## 🚀 Installation

### 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Webcam (for collecting images and real-time verification)

### ⚙️ Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Fawzy-AI-Explorer/Deep-Facial-Recognition.git
   cd Deep-Facial-Recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 📓 Jupyter Notebook
The project includes a comprehensive Jupyter notebook (`Deep Facial Recognition.ipynb`) that guides you through:
- Setting up the environment
- Collecting and processing data
- Building and training the Siamese network model
- Evaluating model performance
- Testing face verification

### 💻 GUI Application
The application provides a user interface for face verification:

1. Run the application:
   ```bash
   python APP/app.py
   ```

2. Use the interface to:
   - Capture new facial images
   - Perform face verification against stored images
   - View verification results

### 📸 Dataset Collection
The project includes tools for collecting your own facial dataset:
- Capture anchor, positive, and negative images using your webcam
- Process and organize images for training

## 🏗️ Model Architecture
The system uses a Siamese neural network that processes pairs of facial images and computes their similarity. The architecture:
1. Takes two facial images as input
2. Processes each through identical neural networks with shared weights
3. Computes the distance between resulting embeddings
4. Determines if images are of the same person based on this distance

## 🏋️ Training
The model is trained using contrastive loss, which:
- Minimizes the distance between embeddings of similar images
- Maximizes the distance between embeddings of different images

Training checkpoints are saved at regular intervals in the `training_checkpoints/` directory.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments
- The LFW (Labeled Faces in the Wild) dataset from Kaggle
- PyTorch and Kivy community resources
