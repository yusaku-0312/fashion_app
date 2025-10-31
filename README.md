# AI Fashion Experiment App

被験者がオンラインで衣服画像をアップロードし、AIが好き嫌いの傾向を学習して印象を予測する実験用Webアプリケーション。

## 概要

このアプリケーションは、以下の3つのフェーズで構成されています：

1. **フェーズ 1**: 被験者が好きな衣服5枚をアップロード
2. **フェーズ 2**: 被験者が嫌いな衣服5枚をアップロード
3. **フェーズ 3**: AIが予測した印象に対して、被験者が5段階評価

## 技術スタック

- **バックエンド**: Flask (Python)
- **AI処理**: OpenAI API (gpt-4o-mini)
- **データ保存**: Google Sheets (n8n Webhook経由)
- **ホスティング**: Render

## ディレクトリ構成

```
flask_fashion_app/
├── app.py                 # Flask アプリケーションのメインファイル
├── requirements.txt       # Python 依存パッケージ
├── Procfile              # Render デプロイ設定
├── .env.example          # 環境変数テンプレート
├── static/
│   └── style.css         # CSS スタイルシート
├── templates/
│   ├── index.html        # 好きな服アップロードフォーム
│   ├── second.html       # 嫌いな服アップロードフォーム
│   ├── output.html       # 評価フォーム
│   ├── thanks.html       # 完了メッセージ
│   └── error.html        # エラーページ
├── test_data/
│   ├── img1.jpg ... img15.jpg  # 評価用画像（15枚）
│   └── README.md         # テストデータの説明
└── uploads/              # アップロードされた画像の一時保存先（自動作成）
```

## セットアップ手順

### 1. 環境構築

```bash
# リポジトリをクローン
git clone <repository-url>
cd flask_fashion_app

# Python 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example` をコピーして `.env` ファイルを作成し、以下の値を設定します：

```bash
cp .env.example .env
```

`.env` ファイルを編集：

```
OPENAI_API_KEY=your_openai_api_key_here
N8N_WEBHOOK_LIKE=https://n8n.your-domain.com/webhook/like_form
N8N_WEBHOOK_DISLIKE=https://n8n.your-domain.com/webhook/dislike_form
N8N_WEBHOOK_RESULT=https://n8n.your-domain.com/webhook/result_form
SECRET_KEY=your_secret_key_here
PORT=5000
FLASK_ENV=production
```

### 3. テストデータの配置

`test_data/` ディレクトリに評価用の衣服画像15枚を配置します：

```
test_data/
├── img1.jpg
├── img2.jpg
├── ...
└── img15.jpg
```

詳細は `test_data/README.md` を参照してください。

### 4. ローカル開発環境での実行

```bash
python app.py
```

ブラウザで `http://localhost:5000` にアクセスします。

## API 連携

### OpenAI API

**用途**: 衣服画像から判断基準を抽出し、新しい衣服に対する印象を予測

**モデル**: `gpt-4o-mini`

**プロンプト**:
- 好きな服の判断基準抽出
- 嫌いな服の判断基準抽出
- 印象予測（提案手法・比較手法）

### n8n Webhook

**用途**: Flask からデータを n8n に送信し、Google Sheets に保存

**Webhook エンドポイント**:
1. `N8N_WEBHOOK_LIKE`: 好きな服の判断基準を送信
2. `N8N_WEBHOOK_DISLIKE`: 嫌いな服の判断基準を送信
3. `N8N_WEBHOOK_RESULT`: 評価結果を送信

**送信データ形式**:

```json
{
  "account_name": "user001",
  "timestamp": "2024-01-01T12:00:00",
  "like_criteria": "・〜〜〜\n・〜〜〜\n..."
}
```

## n8n 側の設定

### Webhook ノード

各 Webhook に対して以下の設定を行います：

1. **Webhook ノード**: POST メソッドでデータを受信
2. **Set ノード**: 受信データにタイムスタンプを追加し、データ構造を整形
3. **Google Sheets ノード**: 指定されたシートにデータを追記
4. **Respond to Webhook ノード**: ステータス `200 OK` を返却

### Google Sheets シート構成

**File 1 (DB1)**:
- `LIKE` シート: 好きな服の判断基準
- `DISLIKE` シート: 嫌いな服の判断基準

**File 2 (DB2)**:
- `OUTPUT` シート: 予測結果とユーザー評価データ

## Render へのデプロイ

### 1. Render アカウントの作成

[Render](https://render.com) にサインアップします。

### 2. 新しい Web Service を作成

1. Render ダッシュボードで「New +」をクリック
2. 「Web Service」を選択
3. GitHub リポジトリを接続
4. 以下の設定を行う：
   - **Name**: `ai-fashion-experiment`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

### 3. 環境変数を設定

Render ダッシュボードの「Environment」セクションで以下を設定：

```
OPENAI_API_KEY=your_openai_api_key
N8N_WEBHOOK_LIKE=https://n8n.your-domain.com/webhook/like_form
N8N_WEBHOOK_DISLIKE=https://n8n.your-domain.com/webhook/dislike_form
N8N_WEBHOOK_RESULT=https://n8n.your-domain.com/webhook/result_form
SECRET_KEY=your_secret_key
FLASK_ENV=production
```

### 4. デプロイ

設定完了後、Render が自動的にデプロイを開始します。

## エラーハンドリング

### OpenAI API エラー

- API キーが無効な場合、エラーメッセージを表示
- API レート制限に達した場合、ユーザーに再試行を促す
- 画像処理エラーの場合、該当の画像をスキップ

### n8n Webhook エラー

- Webhook URL が無効な場合、ログに記録
- タイムアウトが発生した場合、ログに記録（ユーザーには成功と表示）
- 非同期処理のため、失敗時も処理を続行

### ファイルアップロードエラー

- ファイル形式が無効な場合、エラーメッセージを表示
- ファイルサイズが大きすぎる場合（>10MB）、エラーメッセージを表示
- 必須ファイルが不足している場合、エラーメッセージを表示

## ローカルテスト

### テスト用の画像を生成

```bash
# Python を使用してテスト用の画像を生成
python3 << 'EOF'
from PIL import Image, ImageDraw
import os

os.makedirs('test_data', exist_ok=True)

for i in range(1, 16):
    img = Image.new('RGB', (400, 400), color=(73, 109, 137))
    draw = ImageDraw.Draw(img)
    draw.text((150, 190), f"Image {i}", fill=(255, 255, 255))
    img.save(f'test_data/img{i}.jpg')

print("Test images created successfully!")
EOF
```

### 手動テスト

1. ローカル開発環境を起動
2. ブラウザで `http://localhost:5000` にアクセス
3. 好きな服5枚をアップロード
4. 嫌いな服5枚をアップロード
5. 評価フォームで各画像を評価
6. 完了メッセージが表示されることを確認

## トラブルシューティング

### OpenAI API エラー: "Invalid API key"

**解決方法**: `.env` ファイルで `OPENAI_API_KEY` が正しく設定されているか確認

### n8n Webhook エラー: "Connection refused"

**解決方法**: n8n インスタンスが起動しているか、Webhook URL が正しいか確認

### ファイルアップロードエラー: "File too large"

**解決方法**: ファイルサイズを 10MB 以下に圧縮

## セキュリティに関する注意

- `.env` ファイルは **絶対に Git にコミットしないでください**
- `SECRET_KEY` は本番環境で強力なランダム値に変更してください
- OpenAI API キーと n8n Webhook URL は安全に管理してください
- ユーザーがアップロードした画像は、実験終了後に削除してください

## ライセンス

このプロジェクトはプロプライエタリです。

## サポート

ご質問やバグ報告は、プロジェクトメンテナーまでお問い合わせください。

---

**最終更新**: 2024年1月
**バージョン**: 1.0.0

