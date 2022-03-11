# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLIP model for text and image embeddings
model url: https://huggingface.co/openai/clip-vit-base-patch32
"""
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional
import transformers
from PIL import Image
from torch import nn
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPModel(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", processor_name=None):
        super(CLIPModel, self).__init__()

        if processor_name is None:
            processor_name = model_name

        self.model = transformers.CLIPModel.from_pretrained(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)

    def __str__(self):
        return f"CLIPModel({self.model})"

    def forward(self, features):
        image_embeds = []
        text_embeds = []

        if 'pixel_values' in features:
            vision_outputs = self.model.vision_model(pixel_values=features['pixel_values'])
            image_embeds = self.model.visual_projection(vision_outputs[1])

        if 'input_ids' in features:
            text_outputs = self.model.text_model(
                input_ids=features.get('input_ids'),
                attention_mask=features.get('attention_mask', None),
                position_ids=features.get('position_ids', None),
                output_attentions=features.get('output_attentions', None),
                output_hidden_states=features.get('output_hidden_states', None),
            )
            text_embeds = self.model.text_projection(text_outputs[1])

        sentence_embedding = []
        image_features = iter(image_embeds)
        text_features = iter(text_embeds)

        for idx, input_type in enumerate(features['image_text_info']):
            if input_type == 0:
                sentence_embedding.append(next(image_features))
            else:
                sentence_embedding.append(next(text_features))

        features['embedding'] = torch.stack(sentence_embedding).float()

        return features

    def tokenize(self, texts):
        images = []
        texts_values = []
        image_text_info = []

        for idx, data in enumerate(texts):
            if isinstance(data, Image.Image):  # An Image
                images.append(data)
                image_text_info.append(0)
            else:  # A text
                texts_values.append(data)
                image_text_info.append(1)

        if len(texts_values) == 0:
            texts_values = None
        if len(images) == 0:
            images = None

        inputs = self.processor(text=texts_values, images=images, return_tensors="pt", padding=True)
        inputs['image_text_info'] = image_text_info
        return inputs

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    @staticmethod
    def load(input_path: str):
        return CLIPModel(model_name=input_path)

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    @staticmethod
    def batch_to_device(batch):
        """
        send a pytorch batch to a device (CPU/GPU)
        """
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = False
    ):
        """
        Computes sentence and images embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
            In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True
        self.model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sent) for sent in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = self.batch_to_device(features)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features['embedding']
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        else:
            all_embeddings = torch.stack(all_embeddings)
        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings
