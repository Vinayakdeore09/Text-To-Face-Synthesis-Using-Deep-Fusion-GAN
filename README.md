🔍 Criminal Portrait Synthesis from Text to Image Using Deep Learning

This project focuses on synthesizing photorealistic criminal portraits from textual descriptions using deep learning techniques. Leveraging the Deep Fusion GAN (DF-GAN) architecture and Sentence-BERT (SBERT) as the text encoder, the system generates realistic face images conditioned on descriptive text. The CelebA dataset is used as the base for training.

📌 Table of Contents

Demo

Overview

Architecture

Dataset

Installation

Usage

Training

Results

Evaluation Metrics

Future Work

Credits

License

🎥 Demo

Generate high-quality criminal face portraits directly from textual descriptions.

🧠 Overview

This system aims to aid law enforcement and forensic departments by generating suspect images from witness or victim descriptions. Our method combines natural language processing and generative adversarial networks to produce accurate and high-resolution human faces from text.

🏗️ Architecture

Components:

Text Encoder: Sentence-BERT for transforming input descriptions into meaningful embeddings.

Image Generator: Deep Fusion GAN (DF-GAN), which enhances generation quality by directly fusing multi-level text features with image features during generation.

Discriminator: Evaluates the realism of generated images and alignment with text features.

📂 Dataset

CelebA Dataset

Contains over 200,000 celebrity images with 40 attribute labels.

Used for mapping textual descriptions to real-world facial attributes.

Download CelebA

Preprocessing:

Resized images to 64x64 or 128x128.

Converted attributes (e.g., "arched eyebrows", "brown hair", etc.) to natural language descriptions.

Tokenized and embedded using SBERT.

## Model Link : https://drive.google.com/drive/folders/15SSLQ7bqfuReb0tdBrN9hv1pI10kppox?usp=sharing

⚙️ Installation

1. Clone the repository

```bash

git clone https://github.com/vinayakdeore09/criminal-portrait-synthesis.git

cd criminal-portrait-synthesis

Create virtual environment

bash

python -m venv venv

source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies

bash

pip install -r requirements.txt

Download the CelebA dataset and place it in the data/celeba directory.

🚀 Usage

Inference from a text prompt:

bash

python generate.py --text "The woman has high cheekbones and brown straight hair. She is smiling and wearing lipstick." --output generated/

Streamlit Interface

bash

streamlit run app.py

🏋️ Training

To train the model:

bash

python train.py --dataset data/celeba --epochs 200 --batch_size 64

Ensure you modify config.yaml for your hyperparameters, paths, and model settings.

📊 Evaluation Metrics

Fréchet Inception Distance (FID): Measures similarity between real and generated images.

LPIPS (Learned Perceptual Image Patch Similarity): Evaluates perceptual similarity.

Text-Image Consistency: Qualitative validation using human judgments and CLIP score.

📸 Results

Text Description	Generated Image

"A young woman with long black hair, arched eyebrows, and a broad smile"	

"A man with a strong jawline, short brown hair, and a stern expression"	

🔮 Future Work

Improve facial diversity using additional datasets.

Fine-tune SBERT for forensic vocabulary.

Extend to 3D face reconstruction.

Add multilingual support for input descriptions.

🙌 Credits

DF-GAN: Deep Fusion GAN for Text-to-Image Synthesis

SBERT: Sentence-BERT

CelebA Dataset

📄 License

This project is licensed under the MIT License
