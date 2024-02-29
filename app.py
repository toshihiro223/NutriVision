import streamlit as st
import requests
from PIL import Image
import os

# FastAPIエンドポイント
# url = 'http://127.0.0.1:8000/vision/'   # Local
url = 'http://54.173.229.170:8000/vision/'

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
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # ファイルポインタをファイルの先頭に戻す
    uploaded_file.seek(0)
    
    img_path = os.path.join('temp', uploaded_file.name)
    with st.spinner('Classifying...'):
        nutrition_info = classify_image(uploaded_file)
        class_name = nutrition_info['class']
        nutrients = nutrition_info['nutrition']

    st.success(f"Classified as: {class_name}")
    st.subheader('Nutrition Information')

    # 栄養情報を表形式で表示
    # st.table(nutrients)

    # または、より詳細な情報をカード形式で表示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Energy", value=f"{nutrients['Energy']} kcal")
    with col2:
        st.metric(label="Protein", value=f"{nutrients['Protein']} g")
    with col3:
        st.metric(label="Fat", value=f"{nutrients['Fat']} g")
    with col4:
        st.metric(label="Carbohydrates", value=f"{nutrients['Carbohydrate']} g")

    

