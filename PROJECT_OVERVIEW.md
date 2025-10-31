# AI Fashion Experiment App - プロジェクト概要

## 1. プロジェクト目的

本プロジェクトは、被験者がオンラインで衣服画像をアップロードし、AIが好き嫌いの傾向を学習して新しい衣服に対する印象を予測する実験用Webアプリケーションです。

### 主な目標

- 被験者から好きな衣服と嫌いな衣服の画像を収集
- OpenAI API を使用して判断基準を自動抽出
- 抽出された基準を組み合わせて新しい衣服の印象を予測
- ユーザーの評価を収集し、AI予測の精度を検証

## 2. システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
│  (HTML Forms + JavaScript + CSS)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                         │
│  - Route Handlers                                            │
│  - Session Management                                        │
│  - File Upload Processing                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐    ┌─────────────┐   ┌──────────┐
   │ OpenAI  │    │ n8n Webhook │   │ Google   │
   │ API     │    │ (Data Relay)│   │ Sheets   │
   │ (gpt-   │    └─────────────┘   │ (Storage)│
   │ 4o-mini)│                      └──────────┘
   └─────────┘
```

## 3. 技術スタック

| 層 | 技術 | 用途 |
| :--- | :--- | :--- |
| **フロントエンド** | HTML5, CSS3, JavaScript | ユーザーインターフェース |
| **バックエンド** | Flask (Python) | API サーバー、ビジネスロジック |
| **AI/ML** | OpenAI API (gpt-4o-mini) | 画像分析、判断基準抽出、印象予測 |
| **データ連携** | n8n Webhook | Flask と Google Sheets 間のデータ転送 |
| **データ保存** | Google Sheets | 判断基準、評価結果の保存 |
| **ホスティング** | Render | アプリケーションのデプロイ |

## 4. 主要機能

### フェーズ 1: 好きな服のアップロード

**エンドポイント**: `/`

**処理フロー**:
1. ユーザーがアカウント名と好きな服5枚を入力
2. Flask が画像を一時保存
3. OpenAI API で好きな服の判断基準を抽出（10個の基準）
4. 判断基準を n8n Webhook 経由で Google Sheets に保存
5. `/second` にリダイレクト

### フェーズ 2: 嫌いな服のアップロード

**エンドポイント**: `/second`

**処理フロー**:
1. ユーザーが嫌いな服5枚をアップロード
2. Flask が画像を一時保存
3. OpenAI API で嫌いな服の判断基準を抽出（10個の基準）
4. 判断基準を n8n Webhook 経由で Google Sheets に保存
5. **評価用画像15枚に対して印象を予測**:
   - 提案手法: 好き・嫌い基準両方を使用
   - 比較手法: 好き基準のみを使用
6. `/output` にリダイレクト

### フェーズ 3: 評価と結果保存

**エンドポイント**: `/output`

**処理フロー**:
1. ユーザーが各画像に対する5段階評価を入力
2. 評価フォームのバリデーション（15件すべて必須）
3. 予測結果と評価を結合
4. データを n8n Webhook 経由で Google Sheets に保存
5. `/thanks-page` にリダイレクト

## 5. データフロー

### 判断基準の抽出

```
User Images (5枚)
    ↓
[OpenAI API]
    ↓
Judgment Criteria (10個)
    ↓
[n8n Webhook]
    ↓
Google Sheets (LIKE/DISLIKE Sheet)
```

### 印象予測

```
Like Criteria + Dislike Criteria + Evaluation Image (15枚)
    ↓
[OpenAI API] × 15
    ↓
Predictions (提案手法・比較手法)
    ↓
User Ratings (5段階評価)
    ↓
[n8n Webhook]
    ↓
Google Sheets (OUTPUT Sheet)
```

## 6. OpenAI API の使用

### モデル

- **モデル**: `gpt-4o-mini`
- **機能**: マルチモーダル（テキスト + 画像）

### プロンプト設計

#### 好きな服の判断基準抽出

```
これらの服は私のお気に入りの服です。これらの服を多角的に分析して、
私が服を選ぶ時の判断基準を10個予測して下さい。
markdown形式での記述を避け、**などのマークを含めないでください。
```

#### 嫌いな服の判断基準抽出

```
これらの服は私が嫌いなデザインの服です。これらの服を多角的に分析して、
嫌いな服と認定するときの判断基準を10個予測して下さい。
```

#### 印象予測

```
##判断基準
###好きな服から抽出された「どんな服を好みであると認定するかの判断基準」
{{ 好きな服の判断基準 }}
###嫌いな服から抽出された「どんな服を嫌いと認定するかの判断基準」
{{ 嫌いな服の判断基準 }}
##指示
上記の判断基準を持つ人が、この衣服画像を見た時にどんな印象を持つか
一人称視点で予測してください。出力は短文で１つだけ簡潔にお願いします。
```

## 7. n8n ワークフロー

### Webhook エンドポイント

| Webhook | 用途 | 保存先 |
| :--- | :--- | :--- |
| `/webhook/like_form` | 好きな服の判断基準 | File 1 - LIKE Sheet |
| `/webhook/dislike_form` | 嫌いな服の判断基準 | File 1 - DISLIKE Sheet |
| `/webhook/result_form` | 評価結果 | File 2 - OUTPUT Sheet |

### ワークフロー構成

各 Webhook ワークフローは以下の構成：

1. **Webhook ノード**: POST リクエストを受信
2. **Set ノード**: データ構造を整形
3. **Google Sheets ノード**: データを追記
4. **Respond to Webhook ノード**: ステータス 200 を返却

## 8. Google Sheets スキーマ

### File 1 (DB1): 判断基準

**LIKE Sheet**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| like_criteria | Text | 好きな服の判断基準 |

**DISLIKE Sheet**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| dislike_criteria | Text | 嫌いな服の判断基準 |

### File 2 (DB2): 評価結果

**OUTPUT Sheet**:
| Column | Type | Description |
| :--- | :--- | :--- |
| account_name | Text | 被験者のアカウント名 |
| timestamp | DateTime | データ保存日時 |
| image_id | Text | 評価用画像のID (img1-img15) |
| prediction_propose | Text | 提案手法による印象予測 |
| prediction_compare | Text | 比較手法による印象予測 |
| user_score | Integer | ユーザーによる5段階評価 |

## 9. エラーハンドリング

### OpenAI API エラー

**対応**:
- API キーが無効な場合: エラーメッセージを表示
- API レート制限: ユーザーに再試行を促す
- 画像処理エラー: 該当画像をスキップ

### n8n Webhook エラー

**対応**:
- 接続エラー: ログに記録、ユーザーには成功と表示
- タイムアウト: ログに記録、ユーザーには成功と表示
- 非同期処理のため、失敗時も処理を続行

### ファイルアップロードエラー

**対応**:
- ファイル形式が無効: エラーメッセージを表示
- ファイルサイズが大きい: エラーメッセージを表示
- 必須ファイルが不足: エラーメッセージを表示

## 10. セキュリティ考慮事項

### API キー管理

- OpenAI API キーは環境変数で管理
- `.env` ファイルは Git にコミットしない
- 本番環境では Render の Secret 機能を使用

### ユーザーデータ保護

- アップロードされた画像は一時保存後に削除
- セッションデータは暗号化
- HTTPS を使用して通信を保護

### 入力検証

- ファイル形式の検証
- ファイルサイズの制限
- フォーム入力のバリデーション

## 11. パフォーマンス最適化

### 画像処理

- Base64 エンコーディングで API に送信
- 大きな画像は自動的にリサイズ（推奨）

### API 呼び出し

- OpenAI API は評価用画像ごとに2回呼び出し（提案・比較手法）
- 合計: 30 回の API 呼び出し（15枚 × 2）

### セッション管理

- セッションデータはサーバーメモリに保存
- 実験完了後にセッションをクリア

## 12. デプロイメント

### 開発環境

```bash
python app.py
```

### 本番環境（Render）

```bash
gunicorn app:app -b 0.0.0.0:$PORT
```

### 環境変数

```
OPENAI_API_KEY=sk-...
N8N_WEBHOOK_LIKE=https://...
N8N_WEBHOOK_DISLIKE=https://...
N8N_WEBHOOK_RESULT=https://...
SECRET_KEY=...
FLASK_ENV=production
```

## 13. 今後の拡張可能性

### 短期

- [ ] ローディング画面の追加
- [ ] エラーメッセージの改善
- [ ] ユーザーガイドの追加

### 中期

- [ ] データベースの統合（MySQL/PostgreSQL）
- [ ] ユーザー認証機能
- [ ] 結果の可視化ダッシュボード

### 長期

- [ ] 複数言語対応
- [ ] モバイルアプリ化
- [ ] リアルタイム分析機能

## 14. トラブルシューティング

詳細は以下のドキュメントを参照してください：

- **セットアップ**: `README.md`
- **n8n 設定**: `N8N_SETUP_GUIDE.md`
- **デプロイメント**: `DEPLOYMENT_GUIDE.md`

## 15. 参考資料

- [Flask ドキュメント](https://flask.palletsprojects.com/)
- [OpenAI API ドキュメント](https://platform.openai.com/docs)
- [n8n ドキュメント](https://docs.n8n.io/)
- [Render ドキュメント](https://render.com/docs)

---

**プロジェクト開始日**: 2024年1月
**最終更新**: 2024年1月
**バージョン**: 1.0.0

