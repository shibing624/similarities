# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""
import gradio as gr
from similarities import Similarity

sim_model = Similarity()


def ai_text(sentence1, sentence2):
    score = sim_model.similarity(sentence1, sentence2)
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentence1, sentence2, score))

    return score


if __name__ == '__main__':
    examples = [
        ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡'],
        ['我在北京打篮球', '我是北京人，我喜欢篮球'],
        ['一个女人在看书。', '一个女人在揉面团'],
        ['一个男人在车库里举重。', '一个人在举重。'],
    ]
    input1 = gr.inputs.Textbox(lines=2, placeholder="Enter First Sentence")
    input2 = gr.inputs.Textbox(lines=2, placeholder="Enter Second Sentence")

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text,
                 inputs=[input1, input2],
                 outputs=[output_text],
                 # theme="grass",
                 title="Chinese Text Matching Model",
                 description="Copy or input Chinese text here. Submit and the machine will calculate the cosine score.",
                 article="Link to <a href='https://github.com/shibing624/similarities' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples
                 ).launch()
