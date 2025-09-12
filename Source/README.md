# CDXLAV-Finetune-CLIP-for-image-retrieval-Gradio

## How to run code locally and use the app ?
*example app deployed on hugging face space:* https://huggingface.co/spaces/anhquanlam/clip-image-search-app-deepfashion-multimodal

*Download the dataset at:* https://huggingface.co/datasets/anhquanlam/clip-deepfashion-multimodal

1. git clone all files in "code to run locally" folder
2. adjust .env.md to .env. create folder Hugging face/ cache.
3. The folder should look like this: <img width="798" height="540" alt="image" src="https://github.com/user-attachments/assets/436af77e-1dbd-407d-b849-8506bf27ba60" />
4. download the dataset at the link above and unzip it, the folder include dataset folder, and embed_data folder
5. Go to .env file and adjust the Paths of the directory (dataset) on your computer (watch the example paths inside .env file). 
6. create virtual environment.
7. open terminal in the app.py directory, activate the virtual env and 'pip install -r requirement.txt'
8. use 'python app.py' and the app will run, the console will give you a link. Go to that link and you will see the app running.

9. scan the image directory to create embeddings file and indexing for faiss, or just go on and search because i've already scanned and put it in the embed_data folder in the huggingface dataset.

