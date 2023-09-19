# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use faiss to search clip embeddings
"""
import base64
import json
import os
from io import BytesIO
from typing import Optional, Sequence, Union

import faiss
import fire
import numpy as np
import pandas as pd
import requests
from PIL import Image
from loguru import logger

from similarities.clip_module import ClipModule
from similarities.utils.util import cos_sim


def load_data(data, header=None, columns=('image_path', 'text'), delimiter='\t'):
    """
    Encoding data_list text
    @param data: list of (image_path, text) or DataFrame or file path
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


def preprocess_image(image_input: Union[str, np.ndarray, bytes]) -> Image.Image:
    """
    Process image input to Image.Image object
    """
    if isinstance(image_input, str):
        if image_input.startswith('http'):
            return Image.open(requests.get(image_input, stream=True).raw)
        else:
            return Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input)
    elif isinstance(image_input, bytes):
        img_data = base64.b64decode(image_input)
        return Image.open(BytesIO(img_data))
    else:
        raise ValueError("Unsupported image input type")


def clip_embedding(
        input_data_or_path: str,
        columns: Optional[Sequence[str]] = ('image_path', 'text'),
        header: Optional[int] = None,
        delimiter: str = '\t',
        image_embeddings_dir: Optional[str] = 'tmp_image_embeddings_dir/',
        text_embeddings_dir: Optional[str] = 'tmp_text_embeddings_dir/',
        embeddings_name: str = 'emb.npy',
        corpus_file: str = 'tmp_data_dir/corpus.csv',
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        batch_size: int = 32,
        enable_image: bool = True,
        enabel_text: bool = False,
        device: Optional[str] = None,
):
    """Embedding text and image with clip model"""
    df = load_data(input_data_or_path, header=header, columns=columns, delimiter=delimiter)
    logger.info(f'Load data success. data num: {len(df)}, top3: {df.head(3)}')
    images = df['image_path'].tolist()
    texts = df['text'].tolist()
    model = ClipModule(model_name=model_name, device=device)
    logger.info(f'Load model success. model: {model_name}')

    # Start the multi processes pool on all available CUDA devices
    if enable_image:
        os.makedirs(image_embeddings_dir, exist_ok=True)
        images = [preprocess_image(img) for img in images]
        image_emb = model.encode(images, batch_size=batch_size, show_progress_bar=True)
        logger.info(f"Embeddings computed. Shape: {image_emb.shape}")
        image_embeddings_file = os.path.join(image_embeddings_dir, embeddings_name)
        np.save(image_embeddings_file, image_emb)
        logger.debug(f"Embeddings saved to {image_embeddings_file}")
    if enabel_text:
        os.makedirs(text_embeddings_dir, exist_ok=True)
        text_emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        logger.info(f"Embeddings computed. Shape: {text_emb.shape}")
        text_embeddings_file = os.path.join(text_embeddings_dir, embeddings_name)
        np.save(text_embeddings_file, text_emb)
        logger.debug(f"Embeddings saved to {text_embeddings_file}")

    # Save corpus
    if corpus_file:
        os.makedirs(os.path.dirname(corpus_file), exist_ok=True)
        df.to_csv(corpus_file, index=False)
        logger.debug(f"data saved to {corpus_file}")


def clip_index(
        image_embeddings_dir: Optional[str] = None,
        text_embeddings_dir: Optional[str] = None,
        image_index_dir: Optional[str] = "tmp_image_index_dir/",
        text_index_dir: Optional[str] = "tmp_text_index_dir/",
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
            raise e
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
            raise e
    else:
        logger.warning(f"Embeddings dir {text_embeddings_dir} not exist")


def _search_index(
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
        inputs = query
        if isinstance(query, str):
            inputs = [query]
        query_features = model.encode(inputs, normalize_embeddings=True)

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
        item = df.iloc[ei].to_dict()
        logger.debug(f"Found: {item}, similarity: {ed}, id: {ei}")
        text_scores.append((item, float(ed), int(ei)))
    # Sort by score desc
    return sorted(text_scores, key=lambda x: x[1], reverse=True)


def clip_filter(
        query,
        output_file: str = "tmp_outputs/result.json",
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'tmp_image_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = 'tmp_data_dir/corpus.csv',
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
):
    """Entry point of clip filter"""
    assert isinstance(query, (np.ndarray, list, str, bytes))

    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    index = faiss.read_index(index_file)
    model = ClipModule(model_name=model_name, device=device)
    df = pd.read_csv(corpus_file)
    logger.info(f'Load model success. model: {model_name}, index: {index}, data size: {len(df)}')
    sorted_text_scores = _search_index(query, model, index, df, num_results, threshold)
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {'query': query,
                 'results': [{'sentence': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                f,
                ensure_ascii=False,
                indent=2
            )
        logger.info(f"Query: {query}, saved result to {output_file}")
    return sorted_text_scores


def clip_server(
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'tmp_image_index_dir/',
        index_name: str = "faiss.index",
        corpus_file: str = 'tmp_data_dir/corpus.csv',
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        port: int = 8002,
):
    """main entry point of clip search backend, start the endpoints"""
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from starlette.middleware.cors import CORSMiddleware

    print("starting boot of clip serve")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    index = faiss.read_index(index_file)
    model = ClipModule(model_name=model_name, device=device)
    df = pd.read_csv(corpus_file)
    logger.info(f'Load model success. model: {model_name}, index: {index}, data size: {len(df)}')

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
            sorted_text_scores = _search_index(q, model, index, df, num_results, threshold)
            result_dict = {'result': sorted_text_scores}
            logger.debug(f"Successfully search done, q:{q}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

    logger.info("Server starting!")
    uvicorn.run(app, host="0.0.0.0", port=port)


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
