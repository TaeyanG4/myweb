# __init__.py

import pandas as pd
import numpy as np
import re
import pickle
import json

from category_encoders import OrdinalEncoder

from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
host_addr = "0.0.0.0"
host_port = 80

# 슬라 서클 비율
def slider_circle_rate(data):
    if data['count_slider'] == 0 or data['count_normal'] == 0:
        return np.nan
    else:
        return data['count_slider'] / data['count_normal']

def encoding(data):
    # 범주형 변수 인코딩 나머지는 전부 object형 그대로
    i = ['total_length', 'hit_length', 'max_combo', 'genre_id', 'language_id', 'beatmap_count']
    data[i] = data[i].astype(int)
    f = ['diff_size', 'diff_overall', 'diff_approach', 'diff_drain', 'count_normal', 'count_slider', 'count_spinner', 'bpm', 'max_combo', 'difficultyrating', 'work_time']
    data[f] = data[f].astype(float)
    b = ['storyboard', 'video']
    data[b] = data[b].astype(bool)

    # 특성공학 및 데이터 전처리 과정; 필요없는 부분은 주석 처리 
    data['slider_circle_rate'] = data.apply(slider_circle_rate, axis=1)

    # 로그스케일링 (모델 성능향상을 위해 데이터의 분포를 정규분포로 만들어주는 과정)
    data["total_length"] = data["total_length"].apply(lambda x: np.log1p(x))
    data["hit_length"] = data["hit_length"].apply(lambda x: np.log1p(x))
    data["count_normal"] = data["count_normal"].apply(lambda x: np.log1p(x))
    data["count_slider"] = data["count_slider"].apply(lambda x: np.log1p(x))
    data["count_spinner"] = data["count_spinner"].apply(lambda x: np.log1p(x))
    data["max_combo"] = data["max_combo"].apply(lambda x: np.log1p(x))
    data["work_time"] = data["work_time"].apply(lambda x: np.log1p(x))

    # 라벨인코딩
    ord = OrdinalEncoder()
    data = ord.fit_transform(data)
    return data

def decoding(data, after_ranked):
    data = pd.DataFrame({
        'y_pred': data,  # 예측값
    })
    data = data.apply(lambda x: np.expm1(x))  # 로그스케일링 되었던 데이터를 원래대로 복구
    output = int(data['y_pred'] * after_ranked)  # 하루평균 플레이수 * 랭크된지 며칠 지났는지
    return output

@app.route('/')
def init():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def index(data=None):
    
    data = None
    output = None

    dict = request.form.to_dict(flat=False)
    data_json = json.dumps(dict)
    data = pd.read_json(data_json)
    data = data[['total_length', 'hit_length', 'diff_size', 'diff_overall',
        'diff_approach', 'diff_drain', 'count_normal', 'count_slider',
        'count_spinner', 'artist', 'title', 'creator', 'creator_id', 'bpm',
        'source', 'genre_id', 'language_id', 'storyboard', 'video', 'max_combo',
        'difficultyrating', 'after_ranked', 'work_time', 'mapper',
        'beatmap_count']] # 컬럼 순서변경
    
    input_data = encoding(data) # 데이터 전처리 과정

    with open('model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)  # 모델 불러오기
    y_pred = model.predict(input_data)  # 예측값 생성
    output = decoding(y_pred, data['after_ranked']) # 예측값 복구
    output = {'output': output}

    return output

if __name__ == "__main__":
    
    app.run(debug=True,host=host_addr,port=host_port)
    
# 89
# 89
# 4.5
# 9
# 9.5
# 5
# 225
# 245
# 0
# ZUTOMAYO
# Time Left (TV Size)
# Bazz B
# 9063995
# 141
# チェンソーマン
# 5
# 3
# 0
# 1
# 745
# 6.01224
# 65
# Bazz B
# 7
# 65