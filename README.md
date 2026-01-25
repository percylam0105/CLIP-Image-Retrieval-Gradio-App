# CLIP-Image-Retrieval-Gradio-App

## 📌 Description
This repository hosts the CLIP Image Search App, a text-to-image and image-to-image retrieval system tailored for the fashion domain. The app leverages a fine-tuned CLIP (Contrastive Language–Image Pretraining) model, specifically adapted using the DeepFashion dataset to enhance alignment between visual and textual representations. Unlike traditional keyword-based search engines, this system captures the nuanced semantics of fashion queries (e.g., "vintage floral dress with puff sleeves") and supports both accurate brute-force cosine similarity search and efficient FAISS-based approximate search (HNSWFlat/IVF). The project includes a user-friendly Gradio interface, a deployed demo on Hugging Face Spaces, a fine-tuned model, dataset, and finetuning notebook.

## Demo & Resources
[![YouTube Project Demo Video](https://img.shields.io/badge/YouTube-Demo_Video-ff0000?logo=youtube)](https://www.youtube.com/watch?v=6h3SuES8a-M)

[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/anhquanlam/clip-image-search-app-deepfashion-multimodal)  
[![Finetuned Model](https://img.shields.io/badge/HuggingFace-Finetuned_Model-blue?logo=huggingface)](https://huggingface.co/anhquanlam/clip-finetuned-deepfashion)  
[![Dataset ZIP](https://img.shields.io/badge/HuggingFace-Dataset-green?logo=huggingface)](https://huggingface.co/datasets/anhquanlam/clip-deepfashion-multimodal/resolve/main/DeepFashion.zip)  

## DEMO PICTURE:

<img width="1919" height="990" alt="image" src="https://github.com/user-attachments/assets/b294222b-1bd7-4b10-bd02-81bee6f878cd" />

<img width="1919" height="989" alt="image" src="https://github.com/user-attachments/assets/39b60ae5-08fb-4797-86f3-23871d39dad7" />

<img width="1917" height="996" alt="image" src="https://github.com/user-attachments/assets/80ebfe3b-0aaf-4a24-919e-e77d90042f36" />

## APPROACH

The project follows a structured pipeline:

Data Preparation: Utilizes a subset of the DeepFashion dataset with image-caption pairs, preprocessed using CLIPProcessor for resizing (224x224), normalization, and caption tokenization (max 77 tokens).

Model Fine-Tuning: Fine-tunes the pre-trained openai/clip-vit-base-patch16 model with AdamW optimizer (learning rate 5×10⁻⁶, batch size 64, 30 epochs) using a Symmetric InfoNCE loss to optimize image-to-text and text-to-image alignment.

Retrieval System: Extracts and stores image embeddings (as .npy files), supports brute-force and FAISS indexing for speed-accuracy trade-offs, and processes queries via CLIP encoders.

Deployment: Integrates a Gradio UI for interactive search and deploys on Hugging Face Spaces with dynamic dataset loading.

## How to Use the Code
*Example app deployed on Hugging Face Space:* https://huggingface.co/spaces/anhquanlam/clip-image-search-app-deepfashion-multimodal

1. Clone the repository and navigate to the Source folder. The directory structure should resemble the example below. <img width="798" height="540" alt="image" src="https://github.com/user-attachments/assets/436af77e-1dbd-407d-b849-8506bf27ba60" />
2. Rename .env.md to .env and create a Huggingface/cache folder. 
3. Download the dataset at: https://huggingface.co/datasets/anhquanlam/clip-deepfashion-multimodal and unzip it, ensuring it includes the images and embed_data folders.
4. Update the .env file with the correct paths on your machine.
5. Create and activate a virtual environment
6. Install dependencies by running pip install -r requirements.txt in the terminal.
7. Launch the app by executing python app.py from the terminal; a local URL (e.g., http://127.0.0.1:7860) will be provided.
8. Access the app via the URL. You can scan the image directory to generate embeddings and FAISS index, or use precomputed embeddings from the embed_data folder to start searching immediately.












