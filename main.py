import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
from pathlib import Path
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from PIL import Image
import uuid
import shutil
import config
from mangum import Mangum


########## 食品画像の認識 ##########
# ラベルの定義
labels = [
    "Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio",
    "Beef tartare", "Beet salad", "Beignets", "Bibimbap",
    "Bread pudding", "Breakfast burrito", "Bruschetta", "Caesar salad",
    "Cannoli", "Caprese salad", "Carrot cake", "Ceviche",
    "Cheesecake", "Cheese plate", "Chicken curry", "Chicken quesadilla",
    "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros",
    "Clam chowder", "Club sandwich", "Crab cakes", "Creme brulee",
    "Croque madame", "Cup cakes", "Deviled eggs", "Donuts",
    "Dumplings", "Edamame", "Eggs benedict", "Escargots",
    "Falafel", "Filet mignon", "Fish and chips", "Foie gras",
    "French fries", "French onion soup", "French toast", "Fried calamari",
    "Fried rice", "Frozen yogurt", "Garlic bread", "Gnocchi",
    "Greek salad", "Grilled cheese sandwich", "Grilled salmon", "Guacamole",
    "Gyoza", "Hamburger", "Hot and sour soup", "Hot dog",
    "Huevos rancheros", "Hummus", "Ice cream", "Lasagna",
    "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons",
    "Miso soup", "Mussels", "Nachos", "Omelette",
    "Onion rings", "Oysters", "Pad thai", "Paella",
    "Pancakes", "Panna cotta", "Peking duck", "Pho",
    "Pizza", "Pork chop", "Poutine", "Prime rib",
    "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake",
    "Risotto", "Samosa", "Sashimi", "Scallops",
    "Seaweed salad", "Shrimp and grits", "Spaghetti bolognese", "Spaghetti carbonara",
    "Spring rolls", "Steak", "Strawberry shortcake", "Sushi",
    "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare",
    "Waffles"
]

# データの前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 水平方向にランダムに反転
    transforms.RandomRotation(10),  # -10度から10度の間でランダムに回転
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色彩の変更
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ネットワークの定義
class FoodNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # self.feature = models.resnet18(pretrained=True)
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.feature = resnet18(weights=weights)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, 101)  # Food101データセットは101クラス

    def forward(self, x):
        h = self.feature(x)
        h = self.dropout(h)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task="multiclass", num_classes=101), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-5)
        return optimizer
      
  
########## 栄養情報を取得 ##########
class NutritionInfoFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = config.BASE_URL

    def get_nutrition_info(self, food_name):
        headers = {'X-Api-Key': self.api_key}
        params = {'query': food_name}
        response = requests.get(self.base_url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            # 必要な栄養情報を抽出
            nutrients = data['foods'][0]['foodNutrients']
            return nutrients
        else:
            return f"Error: Unable to fetch data, Status Code: {response.status_code}"

########## FastAPI ##########
app = FastAPI()

# Lambda用
handler = Mangum(app)

# アップロードされた画像を一時保存する
async def save_upload_file(upload_file: UploadFile, destination_path: Path) -> Path:
    try:
        # aiofilesを使用して非同期にファイルを開き、内容を書き込む
        async with aiofiles.open(destination_path, 'wb') as buffer:
            # アップロードされたファイルの内容を読み込む
            data = await upload_file.read()
            
            # 読み込んだデータをバッファに書き込む
            await buffer.write(data)
            
    except IOError as e:
        # IOErrorはファイルの読み書き操作で発生する可能性があるエラー
        print(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    finally:
        # ファイルオブジェクトを安全にクローズ
        upload_file.file.close()
        
    return destination_path

# トップページ
@app.get('/')
def index():
    return {"NutriVision": 'NutriInfomation'}

# 画像分類と栄養情報の返却
@app.post("/vision/")
# async def classify_image(image_path):
async def classify_image(file: UploadFile = File(...)):
    # 一時ファイルの保存パスを生成（ランダムなファイル名）
    temp_file = Path(f"temp/{uuid.uuid4()}.jpg")
    
    # アップロードされた画像を一時ファイルとして保存
    image_path = await save_upload_file(file, temp_file)
    
    # ネットワークの準備
    food_model = FoodNet().cpu().eval()
    
    # 重みの読み込み
    food_model.load_state_dict(torch.load('models/food_model.pt', map_location=torch.device('cpu')))
    
    # 予測ラベルリストの定義
    predicted_labels = []
    
    # 画像をRGBに変換
    image = Image.open(image_path).convert('RGB')

    # 前処理を適用
    image = transform(image)    

    # バッチ次元の追加
    image = image.unsqueeze(0)  

    # 推論を実行
    with torch.no_grad():  # 勾配計算を無効化
        outputs = food_model(image)  # モデルに画像を通す
        _, predicted = torch.max(outputs, 1)  # 最も高いスコアを持つクラスを取得

    # 推論結果のクラスインデックス
    predicted_class_index = predicted.item()

    # ラベル名の取得
    predicted_label = labels[predicted_class_index]
    predicted_labels.append(predicted_label)
    print(f"predicted_label : {predicted_label}")

    # USDAのAPI Keyを定義
    api_key = config.UDSA_API_KEY
    
    # インスタンス化
    fetcher = NutritionInfoFetcher(api_key)

    # 栄養情報の取得
    for predicted_label in predicted_labels:
      result = fetcher.get_nutrition_info(predicted_label)

      # 栄養情報の初期化（各栄養素にデフォルト値を設定）
      nutrition_info = {
          "class": predicted_label,
          "nutrition": {
              "Energy": None,
              "Protein": None,
              "Carbohydrate": None,
              "Fat": None
          }
      }
      
      # 必要栄養情報のリストを定義
      nutrients_list = [1008, 1003, 1005, 1004]

      # 栄養情報の抽出
      for nutrient in result:
        if nutrient['nutrientId'] == 1008:
            nutrition_info["nutrition"]["Energy"] = nutrient['value']
        elif nutrient['nutrientId'] == 1003:
            nutrition_info["nutrition"]["Protein"] = nutrient['value']
        elif nutrient['nutrientId'] == 1005:
            nutrition_info["nutrition"]["Carbohydrate"] = nutrient['value']
        elif nutrient['nutrientId'] == 1004:
            nutrition_info["nutrition"]["Fat"] = nutrient['value']
      print(nutrition_info)
    return JSONResponse(content=nutrition_info)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
