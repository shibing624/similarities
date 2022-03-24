# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import gradio as gr
from similarities import Similarity

sim_model = Similarity()


def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


sim_model.add_corpus(load_file('data/corpus_100.txt'))


def ai_text(query):
    res = sim_model.most_similar(queries=query, topn=5)
    print(res)
    for q_id, c in res.items():
        print('query:', query)
        print("search top 5:")
        for corpus_id, s in c.items():
            print(f'\t{sim_model.corpus[corpus_id]}: {s:.4f}')
    res_show = '\n'.join(['search top5：'] + [f'text: {sim_model.corpus[corpus_id]} score: {s:.4f}' for corpus_id, s in
                                             list(res.values())[0].items()])
    return res_show


if __name__ == '__main__':
    examples = [
        ['星巴克被嘲笑了'],
        ['西班牙失业率超过50%'],
        ['她在看书'],
        ['一个人弹琴'],
    ]
    input = gr.inputs.Textbox(lines=2, placeholder="Enter Query")
    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Text Semantic Search Model",
                 description="Copy or input Chinese text here. Submit and the machine will find the most similarity texts.",
                 article="Link to <a href='https://github.com/shibing624/similarities' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()
