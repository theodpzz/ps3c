### An Ensemble-based Two-step Framework for Classification of Pep Smear Cell Images 🔬🧬
Official implementation for the challenge paper "PS3C: An Ensemble-based Two-step Framework for Classification of Pep Smear Cell Images".

Submission can be found: 

## Method Overview

Step 1: Models are independently trained for binary classification to predict whether an image is rubbish or not. Final predictions are obtained by averaging the model scores. If the image is classified as non-rubbish, it proceeds to Step 2. Step 2: Models are separately trained for multi-label classification to determine whether the input image contains a healthy cell, an unhealthy cell, or both. Final predictions are computed as the average of model predictions.

<img src="https://github.com/theodpzz/ps3c/blob/main/figures/method_overview.png" alt="Method overview" width="900">

## Getting Started

### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ps3c.git
```

### Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

### Training

To train the model, run the following command:

```bash
python main.py
```

### Acknowledgments

We acknowledge [Kupas et al. 2024](https://www.nature.com/articles/s41597-024-03596-3) for making the APACC dataset available.

## Citation

> [!IMPORTANT]  
> This project is based on the work by Di Piazza et al. If you use this code in your research, please cite the following paper:

```BibTeX
@inproceedings{dipiazza2025ps3c,
  author    = {Di Piazza Theo  and Loic Boussel},
  title     = {PS3C: An Ensemble-based Two-step Framework for Classification of Pep Smear Cell Images},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2025},
  organization = {IEEE},
}
```
