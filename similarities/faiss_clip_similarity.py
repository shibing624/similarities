# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Use faiss to search clip embeddings
"""
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import faiss
import fire
import numpy as np
import pandas as pd
import requests
from PIL import Image
from loguru import logger
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from similarities.clip_module import ClipModule
from similarities.utils.util import cos_sim


def preprocess_image(image_input: Union[str, np.ndarray, bytes]) -> Image.Image:
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


def clip_embedding(
        input_dir: str,
        chunk_size: int = 10000,
        image_embeddings_dir: Optional[str] = 'clip_engine/image_emb/',
        text_embeddings_dir: Optional[str] = 'clip_engine/text_emb/',
        corpus_dir: str = 'clip_engine/corpus/',
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        batch_size: int = 32,
        enable_image: bool = True,
        image_column_name: str = 'image_path',
        enable_text: bool = False,
        text_column_name: str = 'text',
        target_devices: List[str] = None,
        normalize_embeddings: bool = False,
        **kwargs,
):
    """
    Embedding text and image with clip model
    :param input_dir: input dir, support tsv/csv/txt files
    :param chunk_size: chunk size to read the input file
    :param image_embeddings_dir: save image embeddings dir
    :param text_embeddings_dir: save text embeddings dir
    :param corpus_dir: save corpus dir
    :param model_name: clip model name
    :param batch_size: batch size to compute embeddings, default 32
    :param enable_image: enable image embedding, default True
    :param image_column_name: image column name from input file, default image_path
    :param enable_text: enable text embedding, default False
    :param text_column_name: text column name from input file, default text
    :param target_devices: pytorch target devices, e.g. ['cuda:0', cuda:1]
        If None, all available CUDA devices will be used
    :param normalize_embeddings: whether to normalize embeddings before saving
    :param kwargs: read_csv kwargs
    :return: None, save embeddings to image_embeddings_dir and text_embeddings_dir
    """
    assert enable_image or enable_text, "enable_image and enable_text should not be both False"
    input_files = [f for f in os.listdir(input_dir) if f.endswith((".tsv", ".csv", ".txt"))]
    assert len(input_files) > 0, f"input_dir {input_dir} has no tsv/csv/txt files"
    logger.info(f"Start embedding, input files: {input_files}")
    model = ClipModule(model_name_or_path=model_name)
    logger.info(f'Load model success. model: {model_name}')

    for i, file in enumerate(input_files):
        logger.debug(f"Processing file {i + 1}/{len(input_files)}: {file}")
        output_file_counter = 0
        input_path = os.path.join(input_dir, file)
        # Read the input file in chunks
        for chunk_df in pd.read_csv(input_path, chunksize=chunk_size, **kwargs):
            if enable_image:
                images = chunk_df[image_column_name].tolist()

                images = [preprocess_image(img) for img in images]
                pool = model.start_multi_process_pool(target_devices=target_devices)
                # Compute the embeddings using the multi processes pool
                image_emb = model.encode_multi_process(
                    images,
                    pool,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings
                )
                model.stop_multi_process_pool(pool)
                os.makedirs(image_embeddings_dir, exist_ok=True)
                image_embeddings_file = os.path.join(image_embeddings_dir, f"part-{output_file_counter:05d}.npy")
                np.save(image_embeddings_file, image_emb)
                logger.debug(f"Embeddings computed. Shape: {image_emb.shape}, saved to {image_embeddings_file}")
            if enable_text:
                texts = chunk_df[text_column_name].tolist()

                pool = model.start_multi_process_pool(target_devices=target_devices)
                # Compute the embeddings using the multi processes pool
                text_emb = model.encode_multi_process(
                    texts,
                    pool,
                    batch_size=batch_size,
                    normalize_embeddings=normalize_embeddings
                )
                model.stop_multi_process_pool(pool)
                os.makedirs(text_embeddings_dir, exist_ok=True)
                text_embeddings_file = os.path.join(text_embeddings_dir, f"part-{output_file_counter:05d}.npy")
                np.save(text_embeddings_file, text_emb)
                logger.debug(f"Embeddings computed. Shape: {text_emb.shape}, saved to {text_embeddings_file}")

            # Save corpus to Parquet file
            os.makedirs(corpus_dir, exist_ok=True)
            corpus_file = os.path.join(corpus_dir, f"part-{output_file_counter:05d}.parquet")
            chunk_df.to_parquet(corpus_file, index=False)
            logger.debug(f"Corpus data saved to {corpus_file}")
            output_file_counter += 1
    logger.info(f"Embedding done, saved image emb to {image_embeddings_dir} and text emb to {text_embeddings_dir}")


def clip_index(
        image_embeddings_dir: Optional[str] = None,
        text_embeddings_dir: Optional[str] = None,
        image_index_dir: Optional[str] = "clip_engine/image_index/",
        text_index_dir: Optional[str] = "clip_engine/text_index/",
        index_name: str = "faiss.index",
        max_index_memory_usage: str = "4G",
        current_memory_available: str = "16G",
        use_gpu: bool = False,
        nb_cores: Optional[int] = None,
):
    """
    Build indexes from embeddings using autofaiss
    :param image_embeddings_dir: image embeddings dir, required for image search
    :param text_embeddings_dir: optional, text embeddings dir
    :param image_index_dir: folder to save image index dir
    :param text_index_dir: folder to save text index dir
    :param index_name: index name to save, default 'faiss.index'
    :param max_index_memory_usage: Maximum size allowed for the index, this bound is strict
    :param current_memory_available: Memory available on the machine creating the index, having more memory is a boost
        because it reduces the swipe between RAM and disk.
    :param use_gpu: whether to use gpu, default False
    :param nb_cores: Number of cores to use. Will try to guess the right number if not provided
    :return: None, save index to image_index_dir and text_index_dir
    """
    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    if image_embeddings_dir and os.path.exists(image_embeddings_dir):
        logger.debug(f"Starting build index from {image_embeddings_dir}")
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {image_embeddings_dir} ; saving in {image_index_dir}"
        )
        os.makedirs(image_index_dir, exist_ok=True)
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

    if text_embeddings_dir and os.path.exists(text_embeddings_dir):
        logger.debug(f"Starting build index from {text_embeddings_dir}")
        logger.debug(
            f"Embedding path exist, building index "
            f"using embeddings {text_embeddings_dir} ; saving in {text_index_dir}"
        )
        os.makedirs(text_index_dir, exist_ok=True)
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
    result = []
    if isinstance(queries, np.ndarray):
        if queries.size == 0:
            return result
        query_features = queries
    else:
        if not queries:
            return result
        query_features = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)

    if query_features.shape[0] > 0:
        query_features = query_features.astype(np.float32)
        if threshold is not None:
            _, D, I = faiss_index.range_search(query_features, threshold)
        else:
            D, I = faiss_index.search(query_features, num_results)
        for query, d, i in zip(queries, D, I):
            # Sorted faiss search result with distance
            text_scores = []
            for ed, ei in zip(d, i):
                # Convert to json, avoid float values error
                item = df.iloc[ei].to_json(force_ascii=False)
                if debug:
                    logger.debug(f"query: {query}, Found: {item}, similarity: {ed}, id: {ei}")
                text_scores.append((item, float(ed), int(ei)))
            # Sort by score desc
            query_result = sorted(text_scores, key=lambda x: x[1], reverse=True)
            result.append(query_result)
    return result


def clip_filter(
        texts: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        embeddings: Optional[Union[np.ndarray, List[str]]] = None,
        output_file: str = "outputs/result.jsonl",
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'clip_engine/image_index/',
        index_name: str = "faiss.index",
        corpus_dir: str = 'clip_engine/corpus/',
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        debug: bool = False,
):
    """
    Entry point of clip filter, batch search index
    :param texts: optional, list of texts
    :param images: optional, list of image paths or list of image inputs
    :param embeddings: optional, list of embeddings
    :param output_file: output file path, default outputs/result.json
    :param model_name: clip model name
    :param index_dir: index dir, saved by clip_index, default clip_engine/image_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by clip_embedding, default clip_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :param debug: bool, whether to print debug info, default False
    :return: batch search results
    """
    if texts is None and images is None and embeddings is None:
        raise ValueError("must fill one of texts, images and embeddings input")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    faiss_index = faiss.read_index(index_file)
    model = ClipModule(model_name_or_path=model_name, device=device)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(Path(corpus_dir).glob("*.parquet")))
    logger.info(f'Load success. model: {model_name}, index: {faiss_index}, corpus size: {len(df)}')

    queries = None
    if texts is not None and len(texts) > 0:
        queries = texts
        logger.debug(f"Query: texts size {len(texts)}")
    elif images is not None and len(images) > 0:
        queries = [preprocess_image(img) for img in images]
        logger.debug(f"Query: images size {len(images)}")
    elif embeddings is not None:
        queries = embeddings
        if isinstance(queries, list):
            queries = np.array(queries, dtype=np.float32)
        if len(queries.shape) == 1:
            queries = np.expand_dims(queries, axis=0)
        logger.debug(f"Query: embeddings shape {queries.shape}")
    result = batch_search_index(queries, model, faiss_index, df, num_results, threshold, debug=debug)
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            if texts:
                for q, sorted_text_scores in zip(texts, result):
                    json.dump(
                        {'text': q,
                         'results': [{'item': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                        f,
                        ensure_ascii=False,
                    )
                    f.write('\n')
                logger.info(f"Query texts size: {len(texts)}, saved result to {output_file}")
            elif images:
                for q, sorted_text_scores in zip(images, result):
                    json.dump(
                        {'image': q,
                         'results': [{'item': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                        f,
                        ensure_ascii=False,
                    )
                    f.write('\n')
                logger.info(f"Query images size: {len(images)}, saved result to {output_file}")
            elif embeddings is not None:
                for q, sorted_text_scores in zip(queries, result):
                    json.dump(
                        {'emb': q.tolist(),
                         'results': [{'item': i, 'similarity': j, 'id': k} for i, j, k in sorted_text_scores]},
                        f,
                        ensure_ascii=False,
                    )
                    f.write('\n')
                logger.info(f"Query embeddings size: {len(embeddings)}, saved result to {output_file}")
    return result


class Item(BaseModel):
    input: str = Field(..., max_length=512)


class ClipItem(BaseModel):
    text: Optional[str] = Field(None, max_length=512)
    image: Optional[str] = None


class SearchItem(BaseModel):
    text: Optional[str] = Field(None, max_length=512)
    image: Optional[str] = None
    emb: Optional[List[float]] = None


def clip_server(
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        index_dir: str = 'clip_engine/image_index/',
        index_name: str = "faiss.index",
        corpus_dir: str = 'clip_engine/corpus/',
        num_results: int = 10,
        threshold: Optional[float] = None,
        device: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8002,
        debug: bool = False,
):
    """
    Main entry point of clip search backend, start the server
    :param model_name: clip model name
    :param index_dir: index dir, saved by clip_index, default clip_engine/image_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by clip_embedding, default clip_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :param host: server host, default '0.0.0.0'
    :param port: server port, default 8002
    :param debug: bool, whether to print debug info, default False
    :return: None, start the server
    """
    import uvicorn
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware

    logger.info("starting boot of clip server")
    index_file = os.path.join(index_dir, index_name)
    assert os.path.exists(index_file), f"index file {index_file} not exist"
    faiss_index = faiss.read_index(index_file)
    model = ClipModule(model_name_or_path=model_name, device=device)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in sorted(Path(corpus_dir).glob("*.parquet")))
    logger.info(f'Load model success. model: {model_name}, index: {faiss_index}, corpus size: {len(df)}')

    # define the app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/emb')
    async def emb(item: ClipItem):
        try:
            if item.text is not None:
                q = [item.text]
            elif item.image is not None:
                q = [preprocess_image(item.image)]
            else:
                raise ValueError("item should have text or image")
            embeddings = model.encode(q)
            result_dict = {'emb': embeddings.tolist()[0]}
            logger.debug(f"Successfully get embeddings, res shape: {embeddings.shape}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/similarity')
    async def similarity(item1: ClipItem, item2: ClipItem):
        try:
            if item1.text is not None:
                q1 = item1.text
            elif item1.image is not None:
                q1 = preprocess_image(item1.image)
            else:
                raise ValueError("item1 should have text or image")
            if item2.text is not None:
                q2 = item2.text
            elif item2.image is not None:
                q2 = preprocess_image(item2.image)
            else:
                raise ValueError("item2 should have text or image")
            emb1 = model.encode(q1)
            emb2 = model.encode(q2)
            sim_score = cos_sim(emb1, emb2).tolist()[0][0]
            result_dict = {'similarity': sim_score}
            logger.debug(f"Successfully get similarity score, res: {sim_score}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    @app.post('/search')
    async def search(item: SearchItem):
        try:
            if item.text is not None and len(item.text) > 0:
                q = [item.text]
                logger.debug(f"query: text {item.text}")
            elif item.image is not None and len(item.image) > 0:
                q = [preprocess_image(item.image)]
                logger.debug(f"query: image {item.image}")
            elif item.emb is not None:
                q = item.emb
                logger.debug(f"query: emb size {len(item.emb)}")
                if isinstance(q, list):
                    q = np.array(q, dtype=np.float32)
                if len(q.shape) == 1:
                    q = np.expand_dims(q, axis=0)
            else:
                raise ValueError("item should have text or image or emb")
            results = batch_search_index(q, model, faiss_index, df, num_results, threshold, debug=debug)
            # batch search result, input is one, need return the first
            sorted_text_scores = results[0]
            result_dict = {'result': sorted_text_scores}
            logger.debug(f"Successfully search done, res size: {len(sorted_text_scores)}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

    logger.info("Server starting!")
    uvicorn.run(app, host=host, port=port)


class ClipClient:
    def __init__(self, base_url: str = "http://0.0.0.0:8002", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout

    def _post(self, endpoint: str, data: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}/{endpoint}", json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}

    def get_emb(self, text: Optional[str] = None, image: Optional[str] = None) -> List[float]:
        try:
            data = {
                "text": text,
                "image": image,
            }
            response = self._post("emb", data)
            return response.get("emb", [])
        except Exception as e:
            logger.error(e)
            return []

    def get_similarity(self, item1: ClipItem, item2: ClipItem) -> float:
        try:
            data = {"item1": item1.dict(), "item2": item2.dict()}
            response = self._post("similarity", data)
            return response.get("similarity", 0.0)
        except Exception as e:
            logger.error(f"Error: {e}")
            return 0.0

    def search(
            self,
            text: Optional[str] = None,
            image: Optional[str] = None,
            emb: Optional[List[float]] = None
    ):
        try:
            data = {
                "text": text,
                "image": image,
                "emb": emb
            }
            response = self._post("search", data)
            return response.get("result", [])
        except Exception as e:
            logger.error(f"Error: {e}")
            return []


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
