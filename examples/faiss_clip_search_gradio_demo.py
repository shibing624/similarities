# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import base64
import json
import pprint
import sys

import gradio as gr

sys.path.append('..')
from similarities import ClipClient

client = ClipClient("http://0.0.0.0:8002")


def image_path_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        img_str = base64.b64encode(image_file.read()).decode("utf-8")
    return img_str


def search_image(text="", image=None):
    if text and image is not None:
        return "<p>Please provide either text or image input, not both.</p>"

    if image is not None:
        results = client.search(image=image)
    else:
        results = client.search(text=text)

    html_output = ""
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
        html_output += f'Score: {similarity_score:.2f}'
        html_output += f'</div></div>'

    return html_output


def main():
    iface = gr.Interface(
        fn=search_image,
        inputs=[
            gr.inputs.Textbox(lines=3, placeholder="Enter text here..."),
            gr.inputs.Image(type="filepath", label="Upload an image"),
        ],
        outputs=gr.outputs.HTML(label="Search results"),
        title="CLIP Image Search",
        description="Search for similar images using Faiss and Chinese-CLIP. Link to Github: [similarities](https://github.com/shibing624/similarities)",
    )

    iface.launch(server_name='0.0.0.0', server_port=8082)


if __name__ == '__main__':
    main()
