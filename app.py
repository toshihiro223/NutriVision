import streamlit as st
import requests
from PIL import Image
import os
import config
import matplotlib.pyplot as plt
import numpy as np

# FastAPIエンドポイント
url = config.FASTAPI_URL

def classify_image(image):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(url, files=files)
    
    if response.status_code != 200:
      print("Error:", response.text)
    try:
        return response.json()
    except ValueError as e:  # JSON解析エラーのキャッチ
        print("JSON Decode Error:", e)
        # 応答のテキストやステータスコードを基にエラー処理を行う
        return None

st.title('NutriVision')

# 画像アップロード
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # 画像表示幅を500ピクセルに設定
    st.image(image, caption='Uploaded Image.', width=500)
    
    # ファイルポインタをファイルの先頭に戻す
    uploaded_file.seek(0)
    
    img_path = os.path.join('temp', uploaded_file.name)
    with st.spinner('Classifying...'):
        nutrition_info = classify_image(uploaded_file)
        class_name = nutrition_info['class']
        nutrients = nutrition_info['nutrition']

    st.success(f"Classified as: {class_name}")
    st.subheader('Nutrition Information')


    # ドーナツグラフのデータ準備
    labels = ['Protein', 'Fat', 'Carbohydrates']
    sizes = [nutrients['Protein'], nutrients['Fat'], nutrients['Carbohydrate']]
    colors = ['#ff9999','#66b3ff','#99ff99']
    
    # グラフスタイルの調整
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.3))

    # ラベルを線で示す（ドーナツグラフのラベルを非表示に）
    kw = dict(arrowprops=dict(arrowstyle="-", color='white'), zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        # ここで各栄養素の値に単位の"g"を追加
        ax.annotate(f"{labels[i]}: {sizes[i]}g", xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    # 中心の円でドーナツグラフの穴を作成
    centre_circle = plt.Circle((0,0),0.70,fc='black')
    fig.gca().add_artist(centre_circle)

    # エネルギー値を中心に表示
    plt.text(0, 0, f"{nutrients['Energy']} kcal", ha='center', va='center', color='white')
    
    st.pyplot(fig)
    

