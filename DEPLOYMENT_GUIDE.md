# デプロイメントガイド

このドキュメントでは、Flask Fashion Experiment アプリケーションを Render にデプロイする手順を説明します。

## 前提条件

- GitHub アカウント
- Render アカウント
- OpenAI API キー
- n8n インスタンスと Webhook URL

## ステップ 1: GitHub リポジトリの準備

### 1.1 ローカルリポジトリを初期化

```bash
cd /path/to/flask_fashion_app
git init
git add .
git commit -m "Initial commit: Flask Fashion Experiment App"
```

### 1.2 GitHub にプッシュ

```bash
git remote add origin https://github.com/your-username/flask-fashion-app.git
git branch -M main
git push -u origin main
```

## ステップ 2: Render でのセットアップ

### 2.1 Render にサインイン

[Render ダッシュボード](https://dashboard.render.com) にアクセスしてサインインします。

### 2.2 新しい Web Service を作成

1. ダッシュボードで「New +」をクリック
2. 「Web Service」を選択
3. GitHub リポジトリを接続
4. 以下の情報を入力：

| 項目 | 値 |
| :--- | :--- |
| **Name** | `ai-fashion-experiment` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | `Free` (または必要に応じて有料プラン) |

### 2.3 環境変数を設定

Render ダッシュボードの「Environment」セクションで以下を追加：

```
OPENAI_API_KEY=sk-your-openai-api-key
N8N_WEBHOOK_LIKE=https://n8n.your-domain.com/webhook/like_form
N8N_WEBHOOK_DISLIKE=https://n8n.your-domain.com/webhook/dislike_form
N8N_WEBHOOK_RESULT=https://n8n.your-domain.com/webhook/result_form
SECRET_KEY=your-random-secret-key-here
FLASK_ENV=production
```

**SECRET_KEY の生成**:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 2.4 デプロイを開始

設定完了後、Render が自動的にデプロイを開始します。

デプロイの進行状況は「Deployments」タブで確認できます。

## ステップ 3: デプロイ後の確認

### 3.1 アプリケーションの動作確認

Render ダッシュボードで割り当てられた URL にアクセスします。

例: `https://ai-fashion-experiment.onrender.com`

### 3.2 ログの確認

デプロイ中またはデプロイ後にエラーが発生した場合、ログを確認します：

1. Render ダッシュボードで Web Service を選択
2. 「Logs」タブをクリック
3. エラーメッセージを確認

### 3.3 n8n Webhook の接続確認

1. Flask アプリケーションで `/` にアクセス
2. テスト用のアカウント名を入力
3. テスト画像をアップロード
4. n8n のログで Webhook が呼び出されたことを確認

## ステップ 4: カスタムドメインの設定（オプション）

### 4.1 カスタムドメインを追加

1. Render ダッシュボードで Web Service を選択
2. 「Settings」タブをクリック
3. 「Custom Domain」セクションで「Add Custom Domain」をクリック
4. ドメイン名を入力

### 4.2 DNS 設定

ドメインプロバイダーで DNS レコードを設定します：

```
Type: CNAME
Name: ai-fashion-experiment
Value: ai-fashion-experiment.onrender.com
```

## ステップ 5: 本番環境での設定

### 5.1 セキュリティ設定

- `SECRET_KEY` を強力なランダム値に変更
- HTTPS が有効になっていることを確認
- OpenAI API キーを安全に管理

### 5.2 ログ監視

1. Render ダッシュボードで「Logs」を定期的に確認
2. エラーが発生した場合、すぐに対応

### 5.3 バックアップ戦略

- Google Sheets のデータを定期的にバックアップ
- n8n ワークフローの設定をエクスポート

## トラブルシューティング

### デプロイが失敗する

**確認項目**:
1. `requirements.txt` が正しいか
2. `Procfile` が正しいか
3. `app.py` に構文エラーがないか

**解決方法**:
```bash
# ローカルでテスト
python app.py

# または gunicorn でテスト
gunicorn app:app
```

### アプリケーションが起動しない

**確認項目**:
1. 環境変数が正しく設定されているか
2. `OPENAI_API_KEY` が有効か
3. ログにエラーメッセージがないか

### n8n Webhook が呼び出されない

**確認項目**:
1. n8n インスタンスが起動しているか
2. Webhook URL が正しいか
3. n8n ワークフローがアクティベートされているか

**デバッグ方法**:
```bash
# Flask ログで Webhook 送信を確認
curl -X POST https://your-app.onrender.com/api/test \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'
```

## パフォーマンス最適化

### 1. 画像処理の最適化

- 大きな画像は自動的にリサイズ
- キャッシング機構の導入

### 2. API レート制限への対応

- OpenAI API のレート制限に注意
- 必要に応じて待機時間を追加

### 3. セッション管理

- セッションデータの定期的なクリーンアップ
- メモリ使用量の監視

## スケーリング

### 無料プランの制限

- メモリ: 512 MB
- CPU: 共有
- 月間アップタイム: 750 時間

### 有料プランへのアップグレード

トラフィックが増加した場合、Render ダッシュボードで有料プランにアップグレードできます。

## 監視とアラート

### Render の監視機能

1. ダッシュボードで「Metrics」タブを確認
2. CPU、メモリ、ネットワーク使用量を監視

### ログ集約（オプション）

外部ログサービス（例: Datadog, New Relic）を統合することを検討してください。

## 定期メンテナンス

### 週次

- ログの確認
- エラーの有無を確認

### 月次

- 依存パッケージの更新確認
- セキュリティパッチの適用

### 四半期ごと

- パフォーマンス分析
- スケーリング必要性の評価

## 本番環境チェックリスト

- [ ] `SECRET_KEY` が強力なランダム値に変更されている
- [ ] `OPENAI_API_KEY` が有効で、正しく設定されている
- [ ] `N8N_WEBHOOK_*` URL が正しく設定されている
- [ ] HTTPS が有効になっている
- [ ] ログが監視されている
- [ ] バックアップ戦略が実装されている
- [ ] エラーハンドリングが適切に実装されている
- [ ] テストが完了している

## サポートとリソース

- [Render ドキュメント](https://render.com/docs)
- [Flask ドキュメント](https://flask.palletsprojects.com/)
- [OpenAI API ドキュメント](https://platform.openai.com/docs)
- [n8n ドキュメント](https://docs.n8n.io/)

---

**最終更新**: 2024年1月

