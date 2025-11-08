"""
Flask Application for AI Fashion Experiment
被験者実験用AI衣服評価Webアプリケーション
"""

import os
import base64
import json
import requests
import logging
import random
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
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
N8N_WEBHOOK_RESULT = os.getenv('N8N_WEBHOOK_RESULT')

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            model="gpt-4.1-nano",
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
    
    system_prompt = """これらの服の特徴を箇条書きで10個書いてください。出力は箇条書きで、markdown形式の記述を避けてください。"""
    
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
            model="gpt-4.1-nano",
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
        return features
    
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def predict_impression(like_criteria, dislike_criteria, like_features, dislike_features, image_path):
    """
    Predict impression of a clothing image based on extracted criteria and features.
    
    Args:
        like_criteria: Extracted criteria for liked clothes (for proposed method)
        dislike_criteria: Extracted criteria for disliked clothes (for proposed method)
        like_features: Extracted features for liked clothes (for comparison method)
        dislike_features: Extracted features for disliked clothes (for comparison method)
        image_path: Path to the evaluation image
    
    Returns:
        Tuple of (prediction_propose, prediction_compare)
    """
    
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
        return None, None
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None, None
    
    media_type = get_image_media_type(image_path)
    
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
    
    try:
        # Proposed method prediction
        response_propose = client.chat.completions.create(
            model="gpt-4.1-nano",
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
        
        # Comparison method prediction
        response_compare = client.chat.completions.create(
            model="gpt-4.1-nano",
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
        
        logger.info(f"Predicted impression for {image_path}")
        return prediction_propose, prediction_compare
    
    except Exception as e:
        logger.error(f"OpenAI API error during prediction: {e}")
        return None, None


def send_to_n8n(webhook_url, data):
    """
    Send data to n8n webhook.
    
    Args:
        webhook_url: n8n webhook URL
        data: Dictionary to send
    
    Returns:
        Boolean indicating success
    """
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
            session['like_image_paths'] = image_paths
            
            n8n_data = {
                'account_name': account_name,
                'timestamp': datetime.now().isoformat(),
                'like_criteria': like_criteria,
                'like_features': like_features
            }
            send_to_n8n(N8N_WEBHOOK_LIKE, n8n_data)
            
            return redirect(url_for('second'))
        
        except Exception as e:
            logger.error(f"Error processing like images: {e}")
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
            
            # 比較手法用：特徴を抽出
            dislike_features = extract_features_from_images(image_paths)
            
            session['dislike_criteria'] = dislike_criteria
            session['dislike_features'] = dislike_features
            session['dislike_image_paths'] = image_paths
            
            n8n_data = {
                'account_name': account_name,
                'timestamp': datetime.now().isoformat(),
                'dislike_criteria': dislike_criteria,
                'dislike_features': dislike_features
            }
            send_to_n8n(N8N_WEBHOOK_DISLIKE, n8n_data)
            
            evaluation_images = []
            test_data_dir = 'test_data'
            
            if os.path.exists(test_data_dir):
                for i in range(1, 21):
                    img_file = f'test{i}.jpg'
                    img_path = os.path.join(test_data_dir, img_file)
                    
                    if os.path.exists(img_path):
                        try:
                            prediction_propose, prediction_compare = predict_impression(
                                like_criteria, dislike_criteria, like_features, dislike_features, img_path
                            )
                            
                            # ランダムに左右の表示順序を決定
                            show_propose_left = random.choice([True, False])
                            
                            evaluation_images.append({
                                'id': f'test{i}',
                                'filename': img_file,
                                'prediction_propose': prediction_propose or 'エラー',
                                'prediction_compare': prediction_compare or 'エラー',
                                'show_propose_left': show_propose_left,
                                'left_prediction': prediction_propose if show_propose_left else prediction_compare,
                                'right_prediction': prediction_compare if show_propose_left else prediction_propose,
                                'left_method': 'propose' if show_propose_left else 'compare',
                                'right_method': 'compare' if show_propose_left else 'propose'
                            })
                        except Exception as e:
                            logger.error(f"Error predicting for {img_file}: {e}")
                            evaluation_images.append({
                                'id': f'test{i}',
                                'filename': img_file,
                                'prediction_propose': 'エラー',
                                'prediction_compare': 'エラー',
                                'show_propose_left': True,
                                'left_prediction': 'エラー',
                                'right_prediction': 'エラー',
                                'left_method': 'propose',
                                'right_method': 'compare'
                            })
            
            session['evaluation_images'] = evaluation_images
            return redirect(url_for('output'))
        
        except Exception as e:
            logger.error(f"Error processing dislike images: {e}")
            return render_template('second.html', account_name=account_name, error=f'エラーが発生しました: {str(e)}'), 500
    
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
    if request.method == 'POST':
        account_name = session.get('account_name')
        evaluation_images = session.get('evaluation_images', [])
        
        if not account_name or not evaluation_images:
            return redirect(url_for('index'))
        
        scores_left = {}
        scores_right = {}
        
        for img in evaluation_images:
            img_id = img['id']
            score_left = request.form.get(f'score_left_{img_id}')
            score_right = request.form.get(f'score_right_{img_id}')
            
            if not score_left or not score_right:
                return render_template('output.html', evaluation_images=evaluation_images, 
                                     error='すべての画像に対して評価を入力してください'), 400
            try:
                scores_left[img_id] = int(score_left)
                scores_right[img_id] = int(score_right)
            except ValueError:
                return render_template('output.html', evaluation_images=evaluation_images, 
                                     error='無効な評価値です'), 400
        
        results = []
        for img in evaluation_images:
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
        
        session.clear()
        return redirect(url_for('thanks_page'))
    
    account_name = session.get('account_name')
    evaluation_images = session.get('evaluation_images', [])
    
    if not account_name or not evaluation_images:
        return redirect(url_for('index'))
    
    return render_template('output.html', evaluation_images=evaluation_images)


@app.route('/thanks-page')
def thanks_page():
    """Route for displaying completion message."""
    return render_template('thanks.html')


@app.route('/test_data/<filename>')
def serve_test_image(filename):
    """Serve test data images."""
    from flask import send_from_directory
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