# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use faiss to search clip embeddings
"""
import base64
import json
import os
import sys
from io import BytesIO
from shutil import copytree
from typing import List
from typing import Optional

import cv2
import faiss
import fire
import numpy as np
import pandas as pd
import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from starlette.middleware.cors import CORSMiddleware
from tqdm import tqdm


def load_data(data, header=None, columns=('image_path', 'text'), delimiter='\t'):
    """
    Encoding data_list text
    @param data: list of (image_path, text)
    @param header: read_csv header
    @param columns: read_csv names
    @param delimiter: read_csv sep
    @return: data_df
    """
    if isinstance(data, list):
        data_df = pd.DataFrame(data, columns=columns)
    elif isinstance(data, str) and os.path.exists(data):
        data_df = pd.read_csv(data, header=header, delimiter=delimiter, names=columns)
    elif isinstance(data, pd.DataFrame):
        data_df = data
    else:
        raise TypeError('should be list or file path')
    return data_df


def is_link(s):
    return s is not None and s.startswith('http')


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        with open(image_file, 'rb') as f:
            img_str = f.read()
            img = img_decode(img_str)
        if img is None:
            try:
                buf = BytesIO()
                image = BytesIO(img_str)
                im = Image.open(image)
                rgb = im.convert('RGB')
                rgb.save(buf, 'jpeg')
                buf.seek(0)
                image_bytes = buf.read()
                data_base64 = str(base64.b64encode(image_bytes),
                                  encoding="utf-8")
                image_decode = base64.b64decode(data_base64)
                img_array = np.frombuffer(image_decode, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def preprocess_image(img, alpha_color=(255, 255, 255)):
    """
    preprocess image
    :param img:
    :param alpha_color:
    :return:
    """
    img = check_img(img)
    if img is None:
        return None
    img = alpha_to_color(img, alpha_color)
    return img


def clip_embedding(
        input_data_or_path: str,
        columns: List[str] = ('image_path', 'text'),
        header: int = None,
        delimiter: str = '\t',
        image_embeddings_dir: str = 'image_embeddings_dir/',
        text_embeddings_dir: str = 'text_embeddings_dir/',
        embeddings_name: str = 'emb.npy',
        corpus_file: str = 'corpus.csv',
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        batch_size: int = 32,
        enable_image: bool = True,
        enabel_text: bool = True,
        device: str = None
):
    df = load_data(input_data_or_path, header=header, columns=columns, delimiter=delimiter)
    logger.info(f'Load data success. data num: {len(df)}, top3: {df[:3]}')
    images = df['image_path'].tolist()
    texts = df['text'].tolist()
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    logger.info(f'Load model success. model: {model}')

    # Start the multi processes pool on all available CUDA devices
    if enable_image:
        os.makedirs(image_embeddings_dir, exist_ok=True)
        pool = model.start_multi_process_pool()
        images = [preprocess_image(img) for img in images]
        image_emb = model.encode_multi_process(images, pool, batch_size=batch_size)
        logger.info(f"Embeddings computed. Shape: {image_emb.shape}")
        model.stop_multi_process_pool(pool)
        image_embeddings_file = os.path.join(image_embeddings_dir, embeddings_name)
        np.save(image_embeddings_file, image_emb)
        logger.debug(f"Embeddings saved to {image_embeddings_file}")
    if enabel_text:
        os.makedirs(text_embeddings_dir, exist_ok=True)
        pool = model.start_multi_process_pool()
        text_emb = model.encode_multi_process(texts, pool, batch_size=batch_size)
        logger.info(f"Embeddings computed. Shape: {text_emb.shape}")
        model.stop_multi_process_pool(pool)
        text_embeddings_file = os.path.join(text_embeddings_dir, embeddings_name)
        np.save(text_embeddings_file, text_emb)
        logger.debug(f"Embeddings saved to {text_embeddings_file}")

    # Save corpus
    df.to_csv(corpus_file, index=False)
    logger.debug(f"data saved to {corpus_file}")


def clip_index(
        image_embeddings_dir: str = None,
        text_embeddings_dir: str = None,
        image_index_dir: str = "image_index_dir/",
        text_index_dir: str = "text_index_dir/",
        index_name: str = "faiss.index",
        max_index_memory_usage: str = "4G",
        current_memory_available: str = "16G",
        use_gpu: bool = False,
        nb_cores: Optional[int] = None,
):
    """indexes text embeddings using autofaiss"""
    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    logger.debug(f"Starting build index from {image_embeddings_dir}")
    if image_embeddings_dir and os.path.exists(image_embeddings_dir):
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {image_embeddings_dir} ; saving in {image_index_dir}"
        )
        index_file = os.path.join(image_index_dir, index_name)
        index_infos_path = os.path.join(image_index_dir, index_name + ".json")
        try:
            build_index(
                embeddings=image_embeddings_dir,
                index_path=index_file,
                index_infos_path=index_infos_path,
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
                use_gpu=use_gpu,
            )
            logger.info(f"Index {image_embeddings_dir} done, saved in {index_file}, index infos in {index_infos_path}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Index {image_embeddings_dir} failed, {e}")
    else:
        logger.warning(f"Embeddings dir {image_embeddings_dir} not exist")

    logger.debug(f"Starting build index from {text_embeddings_dir}")
    if text_embeddings_dir and os.path.exists(text_embeddings_dir):
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {text_embeddings_dir} ; saving in {text_index_dir}"
        )
        index_file = os.path.join(text_index_dir, index_name)
        index_infos_path = os.path.join(text_index_dir, index_name + ".json")
        try:
            build_index(
                embeddings=text_embeddings_dir,
                index_path=index_file,
                index_infos_path=index_infos_path,
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
                use_gpu=use_gpu,
            )
            logger.info(f"Index {text_embeddings_dir} done, saved in {index_file}, index infos in {index_infos_path}")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Index {text_embeddings_dir} failed, {e}")
    else:
        logger.warning(f"Embeddings dir {text_embeddings_dir} not exist")


def search_index(
        query,
        model,
        index,
        df,
        num_results,
        threshold
):
    """Search index with text input"""
    # Query embeddings need to be normalized for cosine similarity
    if query.endswith((".png", ".jpg", ".jpeg", ".bmp")) and os.path.isfile(query):
        img = Image.open(query)
        query_features = model.encode([img], normalize_embeddings=True)
    else:
        query_features = model.encode([query], normalize_embeddings=True)

    if threshold is not None:
        _, d, i = index.range_search(query_features, threshold)
        logger.debug(f"Found {i.shape} items with query '{query}' and threshold {threshold}")
    else:
        d, i = index.search(query_features, num_results)
        logger.debug(f"Found {num_results} items with query '{query}'")
        i = i[0]
        d = d[0]

        min_d = min(d)
        max_d = max(d)
        if max_d - min_d < 20:
            logger.debug(f"The minimum distance is {min_d:.2f} and the maximum is {max_d:.2f}")
            logger.debug(
                "You may want to use these numbers to increase your --num_results parameter. "
                "Or use the --threshold parameter."
            )

    # Sorted faiss search result with distance
    text_scores = []
    for ed, ei in zip(d, i):
        item = df.iloc[ei]
        logger.debug(f"Found: {item}, similarity: {ed}, id: {ei}")
        text_scores.append((item, float(ed), int(ei)))
    # Sort by score desc
    return sorted(text_scores, key=lambda x: x[1], reverse=True)


def clip_filter(
        query,
        output_dir: str = "outputs/",
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'image_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = 'corpus.csv',
        num_results: int = 10,
        threshold: float = None,
        device: str = None,
):
    """Entry point of clip filter"""
    assert isinstance(query, (np.ndarray, list, str, bytes))

    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    df = pd.read_csv(corpus_file)
    index_file = os.path.join(index_dir, index_name)
    index = faiss.read_index(index_file)
    logger.info(f'Load model success. model: {model}, index: {index}, data size: {len(df)}')
    os.makedirs(output_dir, exist_ok=True)
    sorted_text_scores = search_index(query, model, index, df, num_results, threshold)
    # Save results
    output_file = os.path.join(output_dir, 'result.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {'query': query,
             'results': [{'item': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
            f,
            ensure_ascii=False,
            indent=2
        )
    logger.info(f"Query: {query}, saved result to {output_file}")


def clip_server(
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'image_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = 'corpus.csv',
        num_results: int = 10,
        threshold: float = None,
        device: str = None,
        port: int = 8002,
):
    """main entry point of clip search backend, start the endpoints"""
    print("starting boot of clip serve")
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    df = pd.read_csv(corpus_file)
    index_file = os.path.join(index_dir, index_name)
    index = faiss.read_index(index_file)
    logger.info(f'Load model success. model: {model}, index: {index}, data size: {len(df)}')

    # define the app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    class Item(BaseModel):
        input: str = Field(..., max_length=512)

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/emb')
    async def emb(item: Item):
        try:
            q = item.input
            embeddings = model.encode(q)
            result_dict = {'emb': embeddings.tolist()}
            logger.debug(f"Successfully get sentence embeddings, q:{q}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/similarity')
    async def similarity(item1: Item, item2: Item):
        try:
            q1 = item1.input
            q2 = item2.input
            emb1 = model.encode(q1)
            emb2 = model.encode(q2)
            sim_score = cos_sim(emb1, emb2)
            result_dict = {'similarity': sim_score}
            logger.debug(f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/search')
    async def search(item: Item):
        try:
            q = item.input
            sorted_text_scores = search_index(q, model, index, df, num_results, threshold)
            result_dict = {'result': sorted_text_scores}
            logger.debug(f"Successfully search done, q:{q}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

    logger.info("Server starting!")
    uvicorn.run(app, host="0.0.0.0", port=port)


def quantize(emb_folder, index_folder, index_name, max_index_memory_usage, current_memory_available, nb_cores):
    """calls autofaiss to build an index"""

    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    try:
        logger.debug(f"starting index {index_name}")
        if os.path.exists(emb_folder):
            logger.debug(
                f"embedding path exist, building index {index_name}"
                f"using embeddings {emb_folder} ; saving in {index_folder}"
            )
            build_index(
                embeddings=emb_folder,
                index_path=index_folder + "/" + index_name + ".index",
                index_infos_path=index_folder + "/" + index_name + ".json",
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
            )
            logger.debug(f"index {index_name} done")
    except Exception as e:  # pylint: disable=broad-except
        logger.exception(f"index {index_name} failed")
        raise e


def clip_index(
        embeddings_folder,
        index_folder,
        max_index_memory_usage="4G",
        current_memory_available="16G",
        copy_metadata=True,
        image_subfolder="img_emb",
        text_subfolder="text_emb",
        nb_cores=None,
):
    """indexes clip embeddings using autofaiss"""
    quantize(
        embeddings_folder + "/" + image_subfolder,
        index_folder,
        "image",
        max_index_memory_usage,
        current_memory_available,
        nb_cores,
    )
    quantize(
        embeddings_folder + "/" + text_subfolder,
        index_folder,
        "text",
        max_index_memory_usage,
        current_memory_available,
        nb_cores,
    )
    if copy_metadata:
        copytree(embeddings_folder + "/metadata", index_folder + "/metadata")


def main():
    """Main entry point"""
    fire.Fire(
        {
            "clip_embedding": clip_embedding,
            "clip_index": clip_index,
            "clip_filter": clip_filter,
            "clip_server": clip_server,
        }
    )


if __name__ == "__main__":
    main()
