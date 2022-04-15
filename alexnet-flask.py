import flask
import pickle
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
# from vit_jax import models
# from vit_jax import checkpoint
# import flax
import PIL
import numpy as np
from predict import predict

# 当前绝对路径
basedir = os.path.abspath(os.path.dirname(__file__))

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return (flask.render_template('upload_more.html'))

    if flask.request.method == 'POST':
        f = request.files.get('file')
        # 获取安全的文件名 正常的文件名
        filename = secure_filename(f.filename)

        # f.filename.rsplit('.', 1)[1] 获取文件的后缀
        # 把文件重命名
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + "." + "JPG"
        print(filename)
        # 保存的目标绝对地址
        file_path = basedir + "/imgs/"
        # 保存文件到目标文件夹
        f.save(file_path + filename)

        # # 加载模型
        # with open(f'AlexNet.pth', 'rb') as f:
        #     model = pickle.load(f)
        #
        # # 读取图片，做预测，返回结果
        model_pth = 'AlexNet.pth'
        img = PIL.Image.open('./imgs/' + filename)
        # img = img.resize((384, 384))
        # logits, = VisionTransformer.call(params, (np.array(img) / 128 - 1)[None, ...])
        # labels = dict(enumerate(open('labels.txt'), start=1))
        # preds = flax.nn.softmax(logits)
        # for idx in preds.argsort()[:-11:-1]:
        #     print(f'{preds[idx]:.5f} : {labels[idx + 1]}', end='')
        #     predict = labels[idx + 1]
        #     break
        class_idx, class_name = predict(img, model_pth)
        return flask.render_template('upload_more.html', result=class_name)


if __name__ == '__main__':
    app.run()
