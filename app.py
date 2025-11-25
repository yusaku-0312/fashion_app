"""
Flask Application for AI Fashion Experiment
被験者実験用AI衣服評価Webアプリケーション
改善案2改良版: 印象文をN8Nに保存し、メモリ上で表示
"""

import os
import base64
import json
import time
import requests
import logging
import random
import uuid
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
N8N_WEBHOOK_LIKE = os.getenv('N8N_WEBHOOK_LIKE')
N8N_WEBHOOK_DISLIKE = os.getenv('N8N_WEBHOOK_DISLIKE')
N8N_WEBHOOK_IMPRESSION = os.getenv('N8N_WEBHOOK_IMPRESSION')
N8N_WEBHOOK_RESULT = os.getenv('N8N_WEBHOOK_RESULT')

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# メモリ上の印象文キャッシュ（セッションIDをキーとする）
impression_cache = {}

# ============================================================================
# Utility Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(file_path):
    """Encode image file to base64 string."""
    try:
        with open(file_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None


def get_image_media_type(filename):
    """Get media type based on file extension."""
    ext = filename.rsplit('.', 1)[1].lower()
    return 'image/jpeg' if ext in ['jpg', 'jpeg'] else 'image/png'


def extract_criteria_from_images(images_paths, criteria_type='like'):
    """
    Extract judgment criteria from clothing images using OpenAI API.
    
    Args:
        images_paths: List of file paths to images
        criteria_type: 'like' or 'dislike'
    
    Returns:
        Extracted criteria as string (bullet points)
    """
    
    if criteria_type == 'like':
        system_prompt = """これらの服は私のお気に入りの服です。これらの服を多角的に分析して、私が服を選ぶ時の判断基準を10個予測して下さい。
markdown形式での記述を避け、**などのマークを含めないでください。
出力形式:
・〜〜〜
・〜〜〜
・〜〜〜
制限:
判断基準以外のテキストは出力しないでください。"""
    else:
        system_prompt = """これらの服は私が嫌いなデザインの服です。これらの服を多角的に分析して、嫌いな服と認定するときの判断基準を10個予測して下さい。
markdown形式での記述を避け、**などのマークを含めないでください。
出力形式:
・〜〜〜
・〜〜〜
・〜〜〜
制限:
判断基準以外のテキストは出力しないでください。"""
    
    # Build image content for API
    image_content = []
    for img_path in images_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image file not found: {img_path}")
            continue
        
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            media_type = get_image_media_type(img_path)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "auto"
                }
            })
    
    if not image_content:
        raise ValueError("No valid images could be processed")
    
    # Call OpenAI API
    if not client:
        raise ValueError("OpenAI client is not initialized. Please set OPENAI_API_KEY.")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        *image_content
                    ]
                }
            ]
        )
        
        criteria = response.choices[0].message.content
        logger.info(f"Extracted {criteria_type} criteria successfully")
        logger.info(f"[{criteria_type.upper()} CRITERIA]:\n{criteria}")
        return criteria
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def extract_features_from_images(images_paths):
    """
    Extract features from clothing images using OpenAI API (for comparison method).
    
    Args:
        images_paths: List of file paths to images
    
    Returns:
        Extracted features as string (bullet points)
    """
    
    system_prompt = """これらの服の特徴を箇条書きで10個書いてください。出力は箇条書きで、markdown形式の記述を避けてください。
    出力形式:
・〜〜〜
・〜〜〜
・〜〜〜"""
    
    # Build image content for API
    image_content = []
    for img_path in images_paths:
        if not os.path.exists(img_path):
            logger.warning(f"Image file not found: {img_path}")
            continue
        
        base64_image = encode_image_to_base64(img_path)
        if base64_image:
            media_type = get_image_media_type(img_path)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "auto"
                }
            })
    
    if not image_content:
        raise ValueError("No valid images could be processed")
    
    # Call OpenAI API
    if not client:
        raise ValueError("OpenAI client is not initialized. Please set OPENAI_API_KEY.")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        *image_content
                    ]
                }
            ]
        )
        
        features = response.choices[0].message.content
        logger.info(f"Extracted features successfully")
        logger.info(f"[FEATURES]:\n{features}")
        return features
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def predict_impression(account_name, like_criteria, dislike_criteria, like_features, dislike_features, image_path, retry_count=3, retry_delay=2):
    """
    Predict impression of a clothing image based on extracted criteria and features.
    生成した印象文はN8Nに保存し、完全な印象文を返す。
    
    Args:
        account_name: Account name for tracking
        like_criteria: Extracted criteria for liked clothes (for proposed method)
        dislike_criteria: Extracted criteria for disliked clothes (for proposed method)
        like_features: Extracted features for liked clothes (for comparison method)
        dislike_features: Extracted features for disliked clothes (for comparison method)
        image_path: Path to the evaluation image
        retry_count: Number of retries on rate limit error
        retry_delay: Delay in seconds between retries
    
    Returns:
        Dictionary with impression data
    """
    
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
        return None
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None
    
    media_type = get_image_media_type(image_path)
    image_name = os.path.basename(image_path)
    
    # 印象文IDを生成
    impression_id = str(uuid.uuid4())
    
    # Prediction with both criteria (proposed method)
    propose_prompt = f"""##判断基準
###好きな服から抽出された「どんな服を好みであると認定するかの判断基準」
{like_criteria}
###嫌いな服から抽出された「どんな服を嫌いと認定するかの判断基準」
{dislike_criteria}
##指示
上記の判断基準はユーザーが実際に好きな服と嫌いな服からLLMによって抽出された判断基準です。
これらのファッションに対する判断基準を持つ人が、この衣服画像を見た時にどんな印象を持つか一人称視点で予測してください。
出力は短文で１つだけ簡潔にお願いします。"""
    
    # Prediction with features (comparison method)
    compare_prompt = f"""##服の特徴
###好きな服から抽出された特徴
{like_features}
###嫌いな服から抽出された特徴
{dislike_features}
##指示
上記の服の特徴はユーザーの実際に好きな服と嫌いな服からLLMによって抽出されたそれらの服の特徴です。
これらの特徴を参考にし、その人がこの衣服画像を見た時にどんな印象を持つか一人称視点で予測してください。
出力は短文で１個簡潔にお願いします。"""
    
    prediction_propose = None
    prediction_compare = None
    has_error = False
    
    # Proposed method prediction with retry
    for attempt in range(retry_count):
        try:
            response_propose = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": propose_prompt},
                            {"type": "image_url",
                             "image_url": {
                                 "url": f"data:{media_type};base64,{base64_image}",
                                 "detail": "auto"
                             }}
                        ]
                    }
                ]
            )
            prediction_propose = response_propose.choices[0].message.content
            logger.info(f"[PROPOSE] {image_name}: {prediction_propose}")
            break
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit hit for propose method on {image_name}, attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {retry_count} attempts for propose method on {image_name}")
                    prediction_propose = 'エラー: レート制限'
                    has_error = True
            else:
                logger.error(f"OpenAI API error during propose prediction for {image_name}: {e}")
                prediction_propose = 'エラー'
                has_error = True
                break
    
    # 提案手法と比較手法の間に待機時間を入れる
    time.sleep(1)
    
    # Comparison method prediction with retry
    for attempt in range(retry_count):
        try:
            response_compare = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": compare_prompt},
                            {"type": "image_url",
                             "image_url": {
                                 "url": f"data:{media_type};base64,{base64_image}",
                                 "detail": "auto"
                             }}
                        ]
                    }
                ]
            )
            prediction_compare = response_compare.choices[0].message.content
            logger.info(f"[COMPARE] {image_name}: {prediction_compare}")
            break
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit hit for compare method on {image_name}, attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {retry_count} attempts for compare method on {image_name}")
                    prediction_compare = 'エラー: レート制限'
                    has_error = True
            else:
                logger.error(f"OpenAI API error during compare prediction for {image_name}: {e}")
                prediction_compare = 'エラー'
                has_error = True
                break
    
    # N8Nに印象文を保存（バックアップ用）
    impression_data = {
        'impression_id': impression_id,
        'account_name': account_name,
        'image_name': image_name,
        'prediction_propose': prediction_propose,
        'prediction_compare': prediction_compare,
        'timestamp': datetime.now().isoformat(),
        'has_error': has_error
    }
    
    if prediction_propose and prediction_compare and not has_error:
        logger.info(f"Successfully predicted impressions for {image_name}, ID: {impression_id}")
    
    # 完全なデータを返す
    return {
        'impression_id': impression_id,
        'image_name': image_name,
        'account_name': account_name,
        'prediction_propose': prediction_propose,
        'prediction_compare': prediction_compare,
        'timestamp': datetime.now().isoformat(),
        'has_error': has_error
    }


def send_to_n8n(webhook_url, data):
    """
    Send data to n8n webhook.
    
    Args:
        webhook_url: n8n webhook URL
        data: Dictionary to send
    
    Returns:
        Boolean indicating success
    """
    if not webhook_url:
        logger.warning(f"N8N webhook URL not configured")
        return False
    
    try:
        response = requests.post(webhook_url, json=data, timeout=10)
        if response.status_code == 200:
            logger.info(f"Successfully sent data to n8n: {webhook_url}")
            return True
        else:
            logger.warning(f"n8n webhook returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send data to n8n: {e}")
        return False


# ============================================================================
# Routes
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route for uploading liked clothing images.
    好きな服のアップロードフォーム
    """
    if request.method == 'POST':
        account_name = request.form.get('account_name', '').strip()
        if not account_name:
            return render_template('index.html', error='アカウント名を入力してください'), 400
        
        uploaded_files = request.files.getlist('like_images')
        if not uploaded_files or len(uploaded_files) < 5:
            return render_template('index.html', error='好きな服を5枚アップロードしてください'), 400
        
        image_paths = []
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        if len(image_paths) < 5:
            return render_template('index.html', error='有効な画像ファイルが5枚に達しません'), 400
        
        try:
            # 提案手法用：判断基準を抽出
            like_criteria = extract_criteria_from_images(image_paths, criteria_type='like')
            
            # 比較手法用：特徴を抽出
            like_features = extract_features_from_images(image_paths)
            
            session['account_name'] = account_name
            session['like_criteria'] = like_criteria
            session['like_features'] = like_features
            
            n8n_data = {
                'account_name': account_name,
                'timestamp': datetime.now().isoformat(),
                'like_criteria': like_criteria,
                'like_features': like_features
            }
            send_to_n8n(N8N_WEBHOOK_LIKE, n8n_data)
            
            return redirect(url_for('second'))
        
        except Exception as e:
            logger.error(f"Error processing like images: {e}", exc_info=True)
            return render_template('index.html', error=f'エラーが発生しました: {str(e)}'), 500
    
    return render_template('index.html')


@app.route('/second', methods=['GET', 'POST'])
def second():
    """
    Route for uploading disliked clothing images.
    嫌いな服のアップロードフォーム
    """
    if request.method == 'POST':
        account_name = session.get('account_name')
        like_criteria = session.get('like_criteria')
        like_features = session.get('like_features')
        
        if not account_name or not like_criteria or not like_features:
            logger.warning("Session data missing in second route")
            return redirect(url_for('index'))
        
        uploaded_files = request.files.getlist('dislike_images')
        
        if not uploaded_files or len(uploaded_files) < 5:
            return render_template('second.html', account_name=account_name, error='嫌いな服を5枚アップロードしてください'), 400
        
        image_paths = []
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        if len(image_paths) < 5:
            return render_template('second.html', account_name=account_name, error='有効な画像ファイルが5枚に達しません'), 400
        
        try:
            # 提案手法用：判断基準を抽出
            dislike_criteria = extract_criteria_from_images(image_paths, criteria_type='dislike')
            logger.info("Dislike criteria extracted successfully")
            
            # 比較手法用：特徴を抽出
            dislike_features = extract_features_from_images(image_paths)
            logger.info("Dislike features extracted successfully")
            
            n8n_data = {
                'account_name': account_name,
                'timestamp': datetime.now().isoformat(),
                'dislike_criteria': dislike_criteria,
                'dislike_features': dislike_features
            }
            send_to_n8n(N8N_WEBHOOK_DISLIKE, n8n_data)
            
            # メモリ上に印象文を保持する配列
            impressions_list = []
            impressions_for_save = []
            test_data_dir = 'test_data'
            
            # test_dataディレクトリの存在確認
            if not os.path.exists(test_data_dir):
                logger.error(f"test_data directory not found at: {os.path.abspath(test_data_dir)}")
                return render_template('second.html', account_name=account_name, 
                                     error='評価用画像ディレクトリが見つかりません'), 500
            
            logger.info(f"test_data directory found at: {os.path.abspath(test_data_dir)}")
            
            # 各画像に対して印象予測
            for i in range(1, 21):
                img_file = f'test{i}.jpg'
                img_path = os.path.join(test_data_dir, img_file)
                
                if os.path.exists(img_path):
                    try:
                        logger.info(f"Processing {img_file}...")
                        impression_data = predict_impression(
                            account_name, like_criteria, dislike_criteria, 
                            like_features, dislike_features, img_path
                        )
                        
                        if impression_data:
                            # ランダムに左右の表示順序を決定
                            show_propose_left = random.choice([True, False])
                            
                            # メモリ上に完全なデータを保持
                            impressions_list.append({
                                'id': f'test{i}',
                                'filename': img_file,
                                'impression_id': impression_data['impression_id'],
                                'prediction_propose': impression_data['prediction_propose'],
                                'prediction_compare': impression_data['prediction_compare'],
                                'show_propose_left': show_propose_left,
                                'has_error': impression_data['has_error']
                            })
                            impressions_for_save.append({
                                'image_name': impression_data['image_name'],
                                'account_name': impression_data['account_name'],
                                'impression_id': impression_data['impression_id'],
                                'prediction_propose': impression_data['prediction_propose'],
                                'prediction_compare': impression_data['prediction_compare'],
                                'has_error': impression_data['has_error']
                            })
                            
                            logger.info(f"Successfully processed {img_file}")
                        
                        # 各画像処理の後に待機時間を追加（レート制限回避）
                        if i < 20:
                            time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error predicting for {img_file}: {e}", exc_info=True)
                        impressions_list.append({
                            'id': f'test{i}',
                            'filename': img_file,
                            'impression_id': 'error',
                            'prediction_propose': 'エラー',
                            'prediction_compare': 'エラー',
                            'show_propose_left': True,
                            'has_error': True
                        })
                else:
                    logger.warning(f"Image not found: {img_path}")
            send_to_n8n(N8N_WEBHOOK_IMPRESSION, {"data": impressions_for_save})
            logger.info(f"Total evaluation images prepared: {len(impressions_list)}")
            
            if len(impressions_list) == 0:
                logger.error("No evaluation images could be processed")
                return render_template('second.html', account_name=account_name, 
                                     error='評価用画像の処理に失敗しました'), 500
            
            # ダミー項目を追加（注意喚起用）
            dummy_item = {
                'id': 'test22',
                'filename': 'virus.png',
                'impression_id': 'test22',
                'prediction_propose': 'ここでは１と入力してください。',
                'prediction_compare': 'ここでは５を入力してください',
                'show_propose_left': True,
                'has_error': False,
                'is_dummy': True
            }
            impressions_list.append(dummy_item)
            
            # リストをシャッフルしてダミー項目の位置をランダムにする
            random.shuffle(impressions_list)
            
            # メモリ上のキャッシュに保存（session['sid']をキーとする）
            cache_key = session.get('cache_key')
            if not cache_key:
                cache_key = str(uuid.uuid4())
                session['cache_key'] = cache_key
            
            impression_cache[cache_key] = impressions_list
            
            logger.info(f"Stored {len(impressions_list)} impressions in memory cache with key: {cache_key}")
            logger.info(f"Redirecting to output page...")
            
            return redirect(url_for('output'))
        
        except Exception as e:
            logger.error(f"Error processing dislike images: {e}", exc_info=True)
            return render_template('second.html', account_name=account_name, 
                                 error=f'エラーが発生しました: {str(e)}'), 500
    
    account_name = session.get('account_name')
    if not account_name:
        return redirect(url_for('index'))
    
    return render_template('second.html', account_name=account_name)


@app.route('/output', methods=['GET', 'POST'])
def output():
    """
    Route for displaying prediction results and evaluation form.
    印象予測結果と評価フォーム
    """
    account_name = session.get('account_name')
    cache_key = session.get('cache_key')
    
    # デバッグ用ログ
    logger.info(f"=== OUTPUT ROUTE ACCESSED ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"Account name: {account_name}")
    logger.info(f"Cache key: {cache_key}")
    
    if not account_name:
        logger.warning("No account_name in session, redirecting to index")
        return redirect(url_for('index'))
    
    if not cache_key or cache_key not in impression_cache:
        logger.warning("No impression data in cache, redirecting to index")
        return redirect(url_for('index'))
    
    # メモリキャッシュから印象文データを取得
    impressions_list = impression_cache[cache_key]
    
    # 表示用に展開
    expanded_images = []
    for img_data in impressions_list:
        show_propose_left = img_data['show_propose_left']
        
        expanded_images.append({
            'id': img_data['id'],
            'filename': img_data['filename'],
            'impression_id': img_data['impression_id'],
            'prediction_propose': img_data['prediction_propose'],
            'prediction_compare': img_data['prediction_compare'],
            'show_propose_left': show_propose_left,
            'left_prediction': img_data['prediction_propose'] if show_propose_left else img_data['prediction_compare'],
            'right_prediction': img_data['prediction_compare'] if show_propose_left else img_data['prediction_propose'],
            'left_method': 'propose' if show_propose_left else 'compare',
            'right_method': 'compare' if show_propose_left else 'propose',
            'is_dummy': img_data.get('is_dummy', False)  # ダミーフラグを追加
        })
    
    logger.info(f"Loaded {len(expanded_images)} impressions from memory cache")
    
    if request.method == 'POST':
        scores_left = {}
        scores_right = {}
        
        # ダミー画像のバリデーション用変数
        dummy_validation_failed = False
        dummy_error_message = ''
        
        for img in expanded_images:
            img_id = img['id']
            score_left = request.form.get(f'score_left_{img_id}')
            score_right = request.form.get(f'score_right_{img_id}')
            
            if not score_left or not score_right:
                return render_template('output.html', evaluation_images=expanded_images, 
                                     error='すべての画像に対して評価を入力してください'), 400
            try:
                scores_left[img_id] = int(score_left)
                scores_right[img_id] = int(score_right)
            except ValueError:
                return render_template('output.html', evaluation_images=expanded_images, 
                                     error='無効な評価値です'), 400
        
        results = []
        for img in expanded_images:
            img_id = img['id']
            
            # 提案手法と比較手法のスコアを正しく振り分ける
            if img['left_method'] == 'propose':
                score_propose = scores_left[img_id]
                score_compare = scores_right[img_id]
            else:
                score_propose = scores_right[img_id]
                score_compare = scores_left[img_id]
            
            results.append({
                'image_id': img_id,
                'impression_id': img['impression_id'],
                'prediction_propose': img['prediction_propose'],
                'prediction_compare': img['prediction_compare'],
                'score_propose': score_propose,
                'score_compare': score_compare,
                'display_order': 'propose_left' if img['show_propose_left'] else 'compare_left'
            })
        
        n8n_data = {
            'account_name': account_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        send_to_n8n(N8N_WEBHOOK_RESULT, n8n_data)
        
        # メモリキャッシュをクリア
        if cache_key in impression_cache:
            del impression_cache[cache_key]
            logger.info(f"Cleared impression cache for key: {cache_key}")
        
        session.clear()
        return redirect(url_for('thanks_page'))
    
    logger.info(f"Rendering output.html with {len(expanded_images)} images")
    return render_template('output.html', evaluation_images=expanded_images)


@app.route('/thanks-page')
def thanks_page():
    """Route for displaying completion message."""
    return render_template('thanks.html')


@app.route('/test_data/<filename>')
def serve_test_image(filename):
    """Serve test data images."""
    return send_from_directory('test_data', filename)


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return render_template('error.html', error='ファイルサイズが大きすぎます(最大10MB)'), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 error."""
    return render_template('error.html', error='ページが見つかりません'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error."""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', error='サーバーエラーが発生しました'), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True)