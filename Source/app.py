from db import SearchMechanism
from clip import CLIPSearcher
from clusterer import ImageIndexer
import os
import gradio as gr
from pathlib import Path
from PIL import Image
import torch
from dotenv import dotenv_values
import json

# Cho phép chạy nếu lỗi OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
env = dotenv_values('.env')

# Đường dẫn thư mục ảnh và index
path = env['DEFAULT_IMAGES_PATH']
index_path = env['INDEX_PATH']
captions_path = env['CAPTIONS_PATH']
os.environ['HUGGINGFACE_HUB_CACHE'] = env['HUGGINGFACE_HUB_CACHE']

# Kiểm tra đường dẫn captions.json
if not os.path.exists(captions_path):
    print(f"Error: captions.json not found at {captions_path}")

# Khởi tạo các thành phần
clip_searcher = CLIPSearcher()
image_indexer = ImageIndexer(index_path)
search_mechanism = SearchMechanism(clip_searcher, image_indexer)

# Hàm xử lý tìm kiếm bằng văn bản


def search_by_text(text, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_text(text, top_k, use_cluster_search)
    if top_k_df is None or top_k_df.empty:
        print("No results found for input text.")
        return [], None, "No results found"
    images = []
    for _, row in top_k_df.iterrows():
        path = row['image_path']
        if os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception as e:
                print(f"Error opening image at {path}: {e}")
        else:
            print(f"Warning: Image not found → {path}")
    print(
        f"search_by_text: Found {len(images)} images, DataFrame shape: {top_k_df.shape}")
    return images, top_k_df, f"Found {len(images)} results"

# Hàm xử lý tìm kiếm bằng ảnh


def search_by_image(image, top_k, use_cluster_search):
    top_k_df = search_mechanism.query_by_image(
        image, top_k, use_cluster_search)
    if top_k_df is None or top_k_df.empty:
        print("No similar images found.")
        return [], None, "No results found"
    images = []
    for _, row in top_k_df.iterrows():
        path = row["image_path"]
        if os.path.exists(path):
            try:
                images.append(Image.open(path))
            except Exception as e:
                print(f"Error opening image at {path}: {e}")
        else:
            print(f"Warning: Image not found → {path}")
    print(
        f"search_by_image: Found {len(images)} images, DataFrame shape: {top_k_df.shape}")
    return images, top_k_df, f"Found {len(images)} results"

# Hàm tìm kiếm tổng hợp


def combined_search(search_type, text, image, top_k, use_cluster_search):
    if search_type == "Text":
        if not text:
            return gr.Warning("Please enter a text query."), None, "Error: No text provided"
        return search_by_text(text, top_k, use_cluster_search)
    else:
        if image is None:
            return gr.Warning("Please upload an image."), None, "Error: No image provided"
        return search_by_image(image, top_k, use_cluster_search)

# Hàm xử lý khi chọn ảnh trong gallery


def get_image_info(evt: gr.SelectData, top_k_df, captions_path):
    print(f"gallery.select triggered, evt.index: {evt.index}")
    if top_k_df is None or top_k_df.empty or evt.index is None:
        print("No valid DataFrame or index")
        return "Select an image to view details", ""

    row = top_k_df.iloc[evt.index]
    score = row.get("cos_sim", row.get("score", "N/A"))
    # Chuyển tensor thành số thực
    if isinstance(score, torch.Tensor):
        score = score.item()
    image_path = row["image_path"]
    image_name = os.path.basename(image_path)

    try:
        with open(captions_path, 'r') as f:
            captions = json.load(f)
        caption = captions.get(image_name, "No caption available")
    except Exception as e:
        caption = f"Error loading caption: {e}"

    print(f"Selected image: {image_name}, Score: {score}, Caption: {caption}")
    return f"{score}", caption


# Quét thư mục ảnh để tạo lại index


def scan_dir(path):
    if path is None or not os.path.exists(path):
        return gr.Info("Path does not exist")
    search_mechanism.scan_directory(Path(path))
    return path


# Tạo giao diện Gradio
with gr.Blocks(css="body { overflow-y: auto !important; }") as webui:
    gr.Markdown("## CLIP Image Search App")

    top_k_df_state = gr.State()

    path = gr.Textbox(label="Path", info="Path to scan images", value=path)
    scan_dir_btn = gr.Button("Scan Directory", variant="primary")

    with gr.Column():
        with gr.Row(equal_height=True):
            search_type = gr.Radio(
                choices=["Text", "Image"], label="Search by", value="Text")
            with gr.Column():
                top_k_slider = gr.Slider(
                    label="Top K", minimum=1, maximum=50, step=1, value=5)
                use_cluster_search = gr.Checkbox(
                    label="Use FAISS cluster search", value=False)
        with gr.Column(visible=True) as text_input:
            text = gr.Textbox(label="Text", placeholder="Enter text to search")
        with gr.Column(visible=False) as image_input:
            image = gr.Image(label="Image")

        def toggle_inputs(search_type):
            return gr.update(visible=search_type == "Text"), gr.update(visible=search_type == "Image")

        search_type.change(toggle_inputs, inputs=[search_type], outputs=[
                           text_input, image_input])

        search_btn = gr.Button("Search", variant="primary")

        status = gr.Textbox(label="Status", value="Ready")

        gallery = gr.Gallery(label="Results", show_label=True,
                             columns=5, rows=2, height="auto", preview=False)
        image_info_score = gr.Textbox(
            label="Similarity Score", value="Select an image to view details")
        image_info_caption = gr.Textbox(
            label="Caption", value="Select an image to view details")

    search_btn.click(
        fn=combined_search,
        inputs=[search_type, text, image, top_k_slider, use_cluster_search],
        outputs=[gallery, top_k_df_state, status]
    )

    gallery.select(
        fn=get_image_info,
        inputs=[top_k_df_state, gr.State(value=captions_path)],
        outputs=[image_info_score, image_info_caption]
    )

    scan_dir_btn.click(
        fn=scan_dir,
        inputs=[path],
        outputs=path
    )

webui.queue()
webui.launch()
