# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import argparse
import base64
import json
import pprint
import sys

import gradio as gr

sys.path.append('..')
from similarities import ClipClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default="http://0.0.0.0:8002", type=str, help='Base url of clip server')
    parser.add_argument('--image_column_name', default="image_path", type=str, help='Image column name to show')
    parser.add_argument('--port', default=8082, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)
    client = ClipClient(args.base_url)

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
            results = client.search(image=image)
            image_src = "data:image/jpeg;base64," + image_path_to_base64(image)
            html_output += f'Query: <img src="{image_src}" width="200" height="200"><br>'
        else:
            results = client.search(text=text)
            html_output += f'Query: {text}<br>'

        html_output += f'Result Size: {len(results)}<br>'
        for result in results:
            item, similarity_score, _ = result
            item_dict = json.loads(item)
            image_path = item_dict.get(args.image_column_name, "")
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

    text_examples = [["狮子"], ["水果"], ["tiger"]]
    image_examples = [["data/image1.png"], ["data/image10.png"], ["data/image13-like-image1.png"]]

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

    demo.queue().launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
