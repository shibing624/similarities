# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import argparse
import json
import sys

import gradio as gr

sys.path.append('..')
from similarities import BertClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', default="http://0.0.0.0:8001", type=str, help='Base url of clip server')
    parser.add_argument('--port', default=8081, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)
    client = BertClient(args.base_url)

    def search_text(query):
        if not query:
            return ''

        results = client.search(query)

        r = []
        for result in results:
            item, similarity_score, idx = result
            item_dict = json.loads(item)
            text = item_dict.get('sentence', "")
            r.append((text, similarity_score))

        res_show = '\n'.join([f'text: {t} score: {s:.4f}' for t, s in r])
        return res_show

    examples = [
        ['星巴克被嘲笑了'],
        ['西班牙失业率超过50%'],
        ['她在看书'],
        ['一个人弹琴'],
    ]
    gr.Interface(
        search_text,
        inputs="text",
        outputs="text",
        theme="soft",
        title="Search for similar texts using Faiss and text2vec",
        description="Copy or input Chinese text here. Submit and the machine will find the most similarity texts.",
        article="Link to <a href='https://github.com/shibing624/similarities' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
