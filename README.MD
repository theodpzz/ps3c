<p align="center">
  <h2 align="center">[ISBI 2025 Challenge] An Ensemble-based Two-step Framework for Classification of Pap Smear Cell Images 🔬🧬</h2>
</p>

✅ Official implementation for the challenge paper "An Ensemble-based Two-step Framework for Classification of Pap Smear Cell Images".

📄 Grand Challenge paper accepted at ISBI 2025: [arXiv preprint](https://arxiv.org/abs/2503.10312)

🏆 Pap Smear Cell Classification Challenge: [Kaggle Competition link](https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge)

### 🧩 Method Overview

Step 1: Models are independently trained for binary classification to predict whether an image is rubbish or not. Final predictions are obtained by averaging the model scores. If the image is classified as non-rubbish, it proceeds to Step 2. 

Step 2: Models are separately trained for multi-label classification to determine whether the input image contains a healthy cell, an unhealthy cell, or both. Final predictions are computed as the average of model predictions.

<img src="https://github.com/theodpzz/ps3c/blob/main/figures/method_overview.png" alt="Method overview" width="900">

### 🚀 Getting Started

#### Clone the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/theodpzz/ps3c.git
```

#### Installation

Make sure you have Python 3 installed. Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

#### Training

To train the step 1, run the following command (example for Swin):

```bash
python main.py train --yaml_file ./step1/configs/train/swinv2.yaml --step 1
```

For the step 2, run the following command:

```bash
python main.py train --yaml_file ./step2/configs/train/swinv2.yaml --step 2
```

#### Demo

Validation and test sets per-class probabilities can be explored in the demo notebook.

### 🔥 Available resources

Prerain weights for Step 1 and Step 2, as well as the per-class final predicted probabilities, are provided in the [🤗 HuggingFace project repository](https://huggingface.co/theodpzz/ps3c).

### 🙌 Acknowledgments

We acknowledge [Kupas et al. 2024](https://www.nature.com/articles/s41597-024-03596-3) for making the [APACC dataset](https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge/data) available.

### 🔗 Citation

> [!IMPORTANT]  
> This project is based on the work by Di Piazza et al. If you use this code in your research, we would appreciate reference to the following paper:

```BibTeX
@inproceedings{dipiazza2025ps3c,
  author    = {Di Piazza Theo  and Loic Boussel},
  title     = {An Ensemble-based Two-step Framework for Classification of Pap Smear Cell Images},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2025},
  organization = {IEEE},
}
```
