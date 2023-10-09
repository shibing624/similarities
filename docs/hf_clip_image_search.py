# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import base64
import glob
import json
import os
import pprint
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import requests
from PIL import Image
from loguru import logger
from tqdm import tqdm

sys.path.append('..')
from similarities.utils.get_file import http_get
from similarities.clip_module import ClipModule


def batch_search_index(
        queries,
        model,
        faiss_index,
        df,
        num_results,
        threshold,
        debug=False,
):
    """
    Search index with image inputs or image paths (batch search)
    :param queries: list of image paths or list of image inputs or texts or embeddings
    :param model: CLIP model
    :param faiss_index: faiss index
    :param df: corpus dataframe
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param debug: bool, whether to print debug info, default True
    :return: search results
    """
    assert queries is not None, "queries should not be None"
    result = []
    if isinstance(queries, np.ndarray):
        query_features = queries
    else:
        query_features = model.encode(queries, normalize_embeddings=True)

    for query, query_feature in zip(queries, query_features):
        query_feature = query_feature.reshape(1, -1)
        if threshold is not None:
            _, d, i = faiss_index.range_search(query_feature, threshold)
            if debug:
                logger.debug(f"Found {i.shape} items with query '{query}' and threshold {threshold}")
        else:
            d, i = faiss_index.search(query_feature, num_results)
            i = i[0]
            d = d[0]
        # Sorted faiss search result with distance
        text_scores = []
        for ed, ei in zip(d, i):
            # Convert to json, avoid float values error
            item = df.iloc[ei].to_json(force_ascii=False)
            if debug:
                logger.debug(f"Found: {item}, similarity: {ed}, id: {ei}")
            text_scores.append((item, float(ed), int(ei)))
        # Sort by score desc
        query_result = sorted(text_scores, key=lambda x: x[1], reverse=True)
        result.append(query_result)
    return result


def preprocess_image(image_input) -> Image.Image:
    """
    Process image input to Image.Image object
    """
    if isinstance(image_input, str):
        if image_input.startswith('http'):
            return Image.open(requests.get(image_input, stream=True).raw)
        elif image_input.endswith((".png", ".jpg", ".jpeg", ".bmp")) and os.path.isfile(image_input):
            return Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type, image path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input)
    elif isinstance(image_input, bytes):
        img_data = base64.b64decode(image_input)
        return Image.open(BytesIO(img_data))
    else:
        raise ValueError(f"Unsupported image input type, image input: {image_input}")


def main():
    text_examples = [["黑猫"], ["坐着的女孩"], ["两只狗拉雪橇"], ["tiger"], ["full Moon"]]
    image_examples = [["photos/YMJ1IiItvPY.jpg"], ["photos/6Fo47c49zEQ.jpg"], ["photos/OM7CvKnhjfs.jpg"],
                      ["photos/lyStEjlKNSw.jpg"], ["photos/mCbo65vkb80.jpg"]]

    # we get about 25k images from Unsplash
    img_folder = 'photos/'
    clip_folder = 'photos/csv/'
    if not os.path.exists(clip_folder) or len(os.listdir(clip_folder)) == 0:
        os.makedirs(img_folder, exist_ok=True)

        photo_filename = 'unsplash-25k-photos.zip'
        if not os.path.exists(photo_filename):  # Download dataset if not exist
            http_get('http://sbert.net/datasets/' + photo_filename, photo_filename)

        # Extract all images
        with zipfile.ZipFile(photo_filename, 'r') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting'):
                zf.extract(member, img_folder)
        df = pd.DataFrame({'image_path': glob.glob(img_folder + '/*'),
                           'image_name': [os.path.basename(x) for x in glob.glob(img_folder + '/*')]})
        os.makedirs(clip_folder, exist_ok=True)
        df.to_csv(f'{clip_folder}/unsplash-25k-photos.csv', index=False)

    index_dir = 'clip_engine_25k/image_index/'
    index_name = "faiss.index"
    corpus_dir = 'clip_engine_25k/corpus/'
    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"

    logger.info("starting boot of clip server")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    faiss_index = faiss.read_index(index_file)
    model = ClipModule(model_name_or_path=model_name)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(Path(corpus_dir).glob("*.parquet")))
    logger.info(f'Load model success. model: {model_name}, index: {faiss_index}, corpus size: {len(df)}')

    def image_path_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            img_str = base64.b64encode(image_file.read()).decode("utf-8")
        return img_str

    def search_image(text="", image=None):
        html_output = ""

        if not text and not image:
            return "<p>Please provide either text or image input.</p>"

        if text and image is not None:
            return "<p>Please provide either text or image input, not both.</p>"

        if image is not None:
            q = [preprocess_image(image)]
            results = batch_search_index(q, model, faiss_index, df, 25, None, debug=False)[0]
            image_src = "data:image/jpeg;base64," + image_path_to_base64(image)
            html_output += f'Query: <img src="{image_src}" width="200" height="200"><br>'
        else:
            q = [text]
            results = batch_search_index(q, model, faiss_index, df, 25, None, debug=False)[0]
            html_output += f'Query: {text}<br>'

        html_output += f'Result Size: {len(results)}<br>'
        for result in results:
            item, similarity_score, _ = result
            item_dict = json.loads(item)
            image_path = item_dict.get("image_path", "")
            tip = pprint.pformat(item_dict)
            if not image_path:
                continue
            if image_path.startswith("http"):
                image_src = image_path
            else:
                image_src = "data:image/jpeg;base64," + image_path_to_base64(image_path)
            html_output += f'<div style="display: inline-block; position: relative; margin: 10px;">'
            html_output += f'<img src="{image_src}" width="200" height="200" title="{tip}">'
            html_output += f'<div style="position: absolute; bottom: 0; right: 0; background-color: rgba(255, 255, 255, 0.7); padding: 2px 5px;">'
            html_output += f'Score: {similarity_score:.4f}'
            html_output += f'</div></div>'

        return html_output

    def reset_user_input():
        return '', None

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">CLIP Image Search</h1>""")
        gr.Markdown(
            "> Search for similar images using Faiss and Chinese-CLIP. Link to Github: [similarities](https://github.com/shibing624/similarities)")
        with gr.Tab("Text"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(lines=2, placeholder="Enter text here...")
            gr.Examples(
                examples=text_examples,
                inputs=[input_text],
            )

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="Upload an image")
            gr.Examples(
                examples=image_examples,
                inputs=[input_image],
            )

        btn_submit = gr.Button(label="Submit")
        output = gr.outputs.HTML(label="Search results")
        btn_submit.click(search_image, inputs=[input_text, input_image],
                         outputs=output, show_progress=True)
        btn_submit.click(reset_user_input, outputs=[input_text, input_image])

    demo.queue().launch()


if __name__ == '__main__':
    main()
