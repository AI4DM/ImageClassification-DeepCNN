# DeepCNN: Comprehensive Deep Learning for Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains comprehensive tutorials and educational materials for Deep Convolutional Neural Networks (CNNs) focusing on Computer Vision, Image Classification, and Image Segmentation tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Course Content](#course-content)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License & Copyright](#license--copyright)

## Overview

This repository provides a hands-on, practical approach to learning deep learning for computer vision, from classical machine learning foundations to advanced deep learning techniques. The materials are designed for both beginners and intermediate practitioners looking to master CNN architectures and their applications.

### Learning Objectives

- **Fundamental Understanding**: Classical ML to Deep Learning progression
- **CNN Architectures**: From basic CNNs to advanced architectures (VGG, ResNet, etc.)
- **Transfer Learning**: Leveraging pre-trained models for specific tasks
- **Image Segmentation**: Flood area segmentation with real-world applications
- **Hyperparameter Tuning**: Systematic optimization techniques
- **End-to-End Implementation**: Complete ML pipelines from data to deployment

## Repository Structure

```
DeepCNN/
├── README.md                           # This comprehensive guide
├── requirements.txt                    # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                        # Git ignore rules
│
├── 📁Day 2 - Image Classification & CNN/
│   ├── 📁Session_1/
│   │   ├── 📁code/
│   │   │   └── Session_1_Classical-Machine-Learning.ipynb
│   │   └── 📁slide/                    # Presentation materials
│   │
│   └── 📁Session_2/
│       ├── 📁code/
│       │   └── Session_2_Convolutional-Neural-Networks.ipynb
│       └── 📁slide/                    # Presentation materials
│
└── 📁Day 3 - Training Deep Models/
    ├── 📁Session_3/
    │   ├── 📁code/
    │   │   └── Session_3_Deep-Learning.ipynb
    │   └── 📁slide/                    # Presentation materials
    │
    └── 📁Session_4/
        ├── 📁code/
        │   ├── Session_4_End2End.ipynb
        │   ├── flood_segmentation_PT.py    # PyTorch implementation
        │   └── flood_segmentation_TF.py    # TensorFlow implementation
        └── 📁slide/                    # Presentation materials
```

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: 
  - Minimum: 8GB RAM
  - Recommended: 16GB+ RAM, GPU with CUDA support
- **Storage**: At least 5GB free space

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AI4DM/ImageClassification-DeepCNN.git
   cd DeepCNN
   ```

2. **Set up environment** (choose one method):

   **Option A: Using conda (Recommended)**
   ```bash
   conda create -n deepcnn python=3.9
   conda activate deepcnn
   ```

   **Option B: Using venv**
   ```bash
   python -m venv deepcnn_env
   source deepcnn_env/bin/activate  # On Windows: deepcnn_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

5. **Start learning**:
   Navigate to `Day 1 - Image Classification & CNN/Session_1/code/` and open the first notebook!

### Google Colab Alternative

**No local setup required!** Run the notebooks directly in Google Colab with free GPU access:

### [Open In Colab](https://colab.research.google.com/)
<!-- (https://colab.research.google.com/assets/colab-badge.svg) -->

#### **Benefits of Using Colab:**
- ✅ **Free GPU/TPU access** - T4, V100, or TPU runtime
- ✅ **Pre-installed libraries** - Most dependencies already available
- ✅ **No setup required** - Start coding immediately
- ✅ **Cloud storage integration** - Google Drive connectivity
- ✅ **Collaborative editing** - Share and collaborate easily

#### **Quick Colab Setup:**

1. **Upload notebooks to Google Drive**:
   - Create a folder in your Google Drive
   - Upload the `.ipynb` files from this repository

2. **Open with Colab**:
   ```python
   # Right-click on any .ipynb file in Google Drive
   # Select "Open with" → "Google Colaboratory"
   ```

3. **Enable GPU (Recommended)**:
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU
   ```

4. **Install additional packages** (if needed):
   ```python
   !pip install scikit-image tqdm
   # Most other packages are pre-installed
   ```

5. **Mount Google Drive** (for datasets):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

#### **Colab-Specific Tips:**
- 📁 **File paths**: Use `/content/drive/MyDrive/` for Google Drive files
- ⏱️ **Session limits**: ~12 hours for free accounts
- 💾 **Save frequently**: Download important model weights
- 🔄 **Runtime reset**: May be needed for memory management

#### **Colab vs Local Setup:**

| Feature | Google Colab | Local Setup |
|---------|--------------|-------------|
| **Setup Time** | 0 minutes | 15-30 minutes |
| **GPU Access** | Free T4/V100 | Requires CUDA setup |
| **Storage** | 15GB free | Unlimited |
| **Runtime** | 12 hours max | Unlimited |
| **Internet** | Required | Optional |
| **Customization** | Limited | Full control |

**💡 Recommendation**: Start with Colab for quick experimentation, then move to local setup for longer projects.


## Environment Setup

### For GPU Support (Optional but Recommended)

**CUDA Setup (NVIDIA GPUs)**:
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**MPS Support (Apple Silicon Macs)**:
```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Development Environment

**VS Code Extensions (Recommended)**:
- Python
- Jupyter
- Python Docstring Generator
- GitLens

**JupyterLab Extensions**:
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Installation

### Automated Installation

```bash
# Clone and set up in one command
git clone https://github.com/AI4DM/ImageClassification-DeepCNN.git && \
cd DeepCNN && \
python -m venv deepcnn_env && \
source deepcnn_env/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt
```

### Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/AI4DM/ImageClassification-DeepCNN.git
cd DeepCNN

# 2. Create virtual environment
python -m venv deepcnn_env
source deepcnn_env/bin/activate  # On Windows: deepcnn_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install requirements
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch, tensorflow, numpy, matplotlib; print('All packages installed successfully!')"
```

### Troubleshooting Installation

**Common Issues**:

1. **Memory errors during installation**:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

2. **Apple Silicon Mac issues**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Windows long path issues**:
   ```bash
   git config --system core.longpaths true
   ```

## Usage

### Running the Notebooks

1. **Sequential Learning Path**:
   ```
   Session 1 → Session 2 → Session 3 → Session 4
   ```

2. **Individual Sessions**:
   ```bash
   # Navigate to specific session
   cd "Day 2 - Image Classification & CNN/Session_1/code/"
   jupyter lab Session_1_Classical-Machine-Learning.ipynb
   ```

3. **Python Scripts**:
   ```bash
   # Run flood segmentation
   cd "Day 3 - Training Deep Models/Session_4/code/"
   python flood_segmentation_PT.py
   ```

### Configuration

**Data Paths**: Update dataset paths in notebooks:
```python
ROOT_DIR = "/path/to/your/dataset"
IMAGES_DIR = os.path.join(ROOT_DIR, "Image")
MASKS_DIR = os.path.join(ROOT_DIR, "Mask")
```

**Model Paths**: Adjust model save/load paths as needed:
```python
torch.save(model.state_dict(), 'your_model.pth')
```

## Course Content

### Day 2: Image Classification & CNN Fundamentals

#### **Session 1: Classical Machine Learning**
- **Topics**: Feature extraction, traditional ML algorithms, data preprocessing
- **Practical**: Implementing classical approaches for image classification
- **Duration**: ~2-3 hours
- **File**: `Session_1_Classical-Machine-Learning.ipynb`

#### **Session 2: Convolutional Neural Networks**
- **Topics**: CNN architecture, convolution operations, pooling layers
- **Practical**: Building first CNN from scratch
- **Duration**: ~2-3 hours
- **File**: `Session_2_Convolutional-Neural-Networks.ipynb`

### Day 3: Training Deep Models

#### **Session 3: Deep Learning Techniques**
- **Topics**: 
  - Normalization techniques (Batch, Instance, Group, Layer)
  - Activation functions (ReLU, LeakyReLU, Swish, GELU)
  - Model architectures (Custom CNN, U-Net)
  - Data augmentation and preprocessing
  - Transfer learning with VGG16
  - **Real hyperparameter tuning** (not simulated!)
- **Practical**: Flood area segmentation with complete training pipeline
- **Duration**: ~3-4 hours
- **File**: `Session_3_Deep-Learning.ipynb`

#### **Session 4: End-to-End Implementation**
- **Topics**: Production-ready models, inference, deployment strategies
- **Practical**: Complete ML pipeline from training to inference
- **Duration**: ~2-3 hours
- **Files**: 
  - `Session_4_End2End.ipynb`
  - `flood_segmentation_PT.py`
  - `flood_segmentation_TF.py`
  - `inference_TF.py`

## Key Features

### 🔬 **Real Hyperparameter Tuning**
- **Actual training runs** (not simulated results)
- **Multiple optimizers**: Adam, AdamW, SGD
- **Learning rate search**: Systematic evaluation
- **Performance visualization**: Real-time results analysis
- **Best configuration selection**: Automated optimization

### 🧠 **Advanced Architectures**
- **Custom CNN**: Built from scratch with modern techniques
- **U-Net**: State-of-the-art segmentation architecture
- **VGG16 Transfer Learning**: Pre-trained model adaptation
- **Progressive training**: Layer-by-layer unfreezing

### 📊 **Comprehensive Analysis**
- **Data visualization**: Dataset statistics and sample analysis
- **Training monitoring**: Loss curves and accuracy tracking
- **Model comparison**: Performance benchmarking
- **Error analysis**: Detailed failure case examination

### 🛠 **Production Ready**
- **Both frameworks**: PyTorch and TensorFlow implementations
- **Model persistence**: Save/load trained models
- **Inference scripts**: Ready-to-use prediction code
- **Optimization**: Speed and memory optimizations

## Datasets

### Supported Datasets

1. **Flood Area Segmentation Dataset**
This dataset can be downloaded from [here](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation).
   - **Task**: Binary segmentation
   - **Format**: RGB images + binary masks
   - **Size**: ~290 samples
   - **Split**: 80% train, 20% validation

2. **Custom Image Classification**
   - **Task**: Multi-class classification
   - **Format**: Standard image folders
   - **Support**: Any image classification dataset

### Data Structure

```
dataset/
├── Image/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Mask/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── metadata.csv
```

### Data Preparation

```python
# Update paths in notebooks
ROOT_DIR = "/path/to/your/dataset"
IMAGES_DIR = os.path.join(ROOT_DIR, "Image")
MASKS_DIR = os.path.join(ROOT_DIR, "Mask")
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

### How to Contribute 🤝

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes**: Implement your improvements
4. **Add tests**: Ensure code quality
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open Pull Request**: Describe your changes

### Contribution Areas

- 📚 **Documentation improvements**
- 🐛 **Bug fixes and optimizations**
- ✨ **New model architectures**
- 📊 **Additional datasets support**
- 🔧 **Tool and utility enhancements**
- 🎨 **Visualization improvements**

## License & Copyright

### Copyright Notice

**© 2025 Parham Kebria. All Rights Reserved.**

This repository is protected by copyright law. Unauthorized reproduction, distribution, modification, or commercial use is strictly prohibited without written permission from the author.

### Permitted Use

- ✅ **Personal learning and education**
- ✅ **Academic research** (with proper citation)
- ✅ **Classroom instruction** (with attribution)

### Prohibited Use

- ❌ **Commercial redistribution**
- ❌ **Modification without permission**
- ❌ **Removal of copyright notices**
- ❌ **Claiming authorship**

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

**Citation Required:** When referencing this work, please cite:

```bibtex
@misc{kebria2025deepcnn,
  title={DeepCNN: Comprehensive Deep Learning for Computer Vision},
  author={Kebria, Parham},
  year={2025},
  url={https://github.com/parhamkebria/DeepCNN}
}
```

### Contact

**For licensing inquiries and collaborations:**
- 🌐 **Website**: [parhamkebria.com](https://parhamkebria.com)
- 📧 **Email**: Contact through website
- 🐙 **GitHub**: [@parhamkebria](https://github.com/parhamkebria)

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

[![GitHub stars](https://img.shields.io/github/stars/parhamkebria/DeepCNN.svg?style=social&label=Star&maxAge=2592000)](https://github.com/AI4DM/ImageClassification-DeepCNN/stargazers/)

</div>
