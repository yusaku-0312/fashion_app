# n8n ワークフロー設定ガイド

このドキュメントでは、Flask アプリケーションと連携するための n8n ワークフロー設定を説明します。

## 概要

n8n は3つの独立したワークフローを管理します：

1. **like_form**: 好きな服の判断基準を受け取り、Google Sheets に保存
2. **dislike_form**: 嫌いな服の判断基準を受け取り、Google Sheets に保存
3. **result_form**: 評価結果を受け取り、Google Sheets に保存

## 前提条件

- n8n インスタンスが起動していること
- Google Sheets API が有効になっていること
- Google Sheets の認証情報が n8n に登録されていること

## ワークフロー 1: like_form

### ステップ 1: Webhook ノードの設定

1. n8n エディタで新しいワークフローを作成
2. **Webhook** ノードを追加
3. 以下の設定を行う：
   - **HTTP Method**: POST
   - **Path**: `/webhook/like_form`
   - **Authentication**: None (または必要に応じて設定)

### ステップ 2: Set ノードでデータを整形

1. **Set** ノードを追加
2. 以下のフィールドを設定：
   - `account_name`: `{{ $json.account_name }}`
   - `timestamp`: `{{ $json.timestamp }}`
   - `like_criteria`: `{{ $json.like_criteria }}`

### ステップ 3: Google Sheets ノードで保存

1. **Google Sheets** ノードを追加
2. 以下の設定を行う：
   - **Authentication**: Google Sheets API 認証
   - **Operation**: Append Row
   - **Spreadsheet**: File 1 (DB1)
   - **Sheet**: LIKE
   - **Columns**: 
     - `account_name`
     - `timestamp`
     - `like_criteria`

### ステップ 4: Webhook レスポンスを返却

1. **Respond to Webhook** ノードを追加
2. 以下の設定を行う：
   - **Status Code**: 200
   - **Response Body**: `{ "status": "success" }`

## ワークフロー 2: dislike_form

### ステップ 1: Webhook ノードの設定

1. 新しいワークフローを作成
2. **Webhook** ノードを追加
3. 以下の設定を行う：
   - **HTTP Method**: POST
   - **Path**: `/webhook/dislike_form`

### ステップ 2: Set ノードでデータを整形

1. **Set** ノードを追加
2. 以下のフィールドを設定：
   - `account_name`: `{{ $json.account_name }}`
   - `timestamp`: `{{ $json.timestamp }}`
   - `dislike_criteria`: `{{ $json.dislike_criteria }}`

### ステップ 3: Google Sheets ノードで保存

1. **Google Sheets** ノードを追加
2. 以下の設定を行う：
   - **Operation**: Append Row
   - **Spreadsheet**: File 1 (DB1)
   - **Sheet**: DISLIKE
   - **Columns**:
     - `account_name`
     - `timestamp`
     - `dislike_criteria`

### ステップ 4: Webhook レスポンスを返却

1. **Respond to Webhook** ノードを追加
2. ステータスコード 200 を返却

## ワークフロー 3: result_form

### ステップ 1: Webhook ノードの設定

1. 新しいワークフローを作成
2. **Webhook** ノードを追加
3. 以下の設定を行う：
   - **HTTP Method**: POST
   - **Path**: `/webhook/result_form`

### ステップ 2: データ処理

結果データは複数の行（15枚の画像分）を含むため、以下のように処理します：

1. **Set** ノードを追加
2. `results` 配列をループ処理するため、**Loop** ノードを追加
3. ループ内で各結果を Google Sheets に追記

### ステップ 3: Google Sheets ノードで保存

1. **Google Sheets** ノードを追加（ループ内）
2. 以下の設定を行う：
   - **Operation**: Append Row
   - **Spreadsheet**: File 2 (DB2)
   - **Sheet**: OUTPUT
   - **Columns**:
     - `account_name`: `{{ $json.account_name }}`
     - `timestamp`: `{{ $json.timestamp }}`
     - `image_id`: `{{ $item(0).image_id }}`
     - `prediction_propose`: `{{ $item(0).prediction_propose }}`
     - `prediction_compare`: `{{ $item(0).prediction_compare }}`
     - `user_score`: `{{ $item(0).user_score }}`

### ステップ 4: Webhook レスポンスを返却

1. **Respond to Webhook** ノードを追加
2. ステータスコード 200 を返却

## Google Sheets の準備

### File 1 (DB1): 判断基準の保存

**LIKE シート**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| like_criteria | Text | 好きな服の判断基準 |

**DISLIKE シート**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| dislike_criteria | Text | 嫌いな服の判断基準 |

### File 2 (DB2): 評価結果の保存

**OUTPUT シート**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| image_id | Text | 評価用画像のID (img1, img2, ...) |
| prediction_propose | Text | 提案手法による印象予測 |
| prediction_compare | Text | 比較手法による印象予測 |
| user_score | Integer | ユーザーによる5段階評価 |

## Webhook URL の確認

各ワークフローをアクティベートした後、Webhook URL を確認します：

1. ワークフローの **Webhook** ノードをクリック
2. **Webhook URL** をコピー
3. Flask の `.env` ファイルに設定：

```
N8N_WEBHOOK_LIKE=https://n8n.your-domain.com/webhook/like_form
N8N_WEBHOOK_DISLIKE=https://n8n.your-domain.com/webhook/dislike_form
N8N_WEBHOOK_RESULT=https://n8n.your-domain.com/webhook/result_form
```

## テスト方法

### cURL でテスト

```bash
# like_form のテスト
curl -X POST https://n8n.your-domain.com/webhook/like_form \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "test_user",
    "timestamp": "2024-01-01T12:00:00",
    "like_criteria": "・シンプルなデザイン\n・明るい色合い"
  }'

# dislike_form のテスト
curl -X POST https://n8n.your-domain.com/webhook/dislike_form \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "test_user",
    "timestamp": "2024-01-01T12:00:00",
    "dislike_criteria": "・派手な柄\n・暗い色"
  }'

# result_form のテスト
curl -X POST https://n8n.your-domain.com/webhook/result_form \
  -H "Content-Type: application/json" \
  -d '{
    "account_name": "test_user",
    "timestamp": "2024-01-01T12:00:00",
    "results": [
      {
        "image_id": "img1",
        "prediction_propose": "この服はあなたの好みに合いそうです",
        "prediction_compare": "シンプルで好みに合いそうです",
        "user_score": 4
      }
    ]
  }'
```

### n8n エディタでテスト

1. ワークフローの **Webhook** ノードをクリック
2. **Test** ボタンをクリック
3. テストペイロードを入力
4. **Send Test Data** をクリック

## トラブルシューティング

### Webhook が動作しない

**確認項目**:
1. ワークフローがアクティベートされているか
2. Webhook URL が正しいか
3. n8n インスタンスが起動しているか

### Google Sheets への書き込みが失敗する

**確認項目**:
1. Google Sheets API 認証が有効か
2. スプレッドシートのシート名が正しいか
3. 列名が正しいか

### データが重複して保存される

**原因**: Webhook が複数回呼び出されている可能性

**解決方法**: 
1. n8n のログを確認
2. Flask 側で重複送信がないか確認
3. 必要に応じて n8n で重複排除ロジックを追加

## セキュリティに関する注意

- Webhook URL は公開されるため、認証を追加することを推奨
- Google Sheets API キーは安全に管理
- n8n インスタンスは HTTPS で保護されていることを確認

## 参考資料

- [n8n Webhook ノード ドキュメント](https://docs.n8n.io/nodes/n8n-nodes-base.webhook/)
- [n8n Google Sheets ノード ドキュメント](https://docs.n8n.io/nodes/n8n-nodes-base.googleSheets/)
- [Google Sheets API ドキュメント](https://developers.google.com/sheets/api)

---

**最終更新**: 2024年1月

