"""
Test script for Flask Fashion Experiment App
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, allowed_file, encode_image_to_base64, get_image_media_type


class FlaskAppTestCase(unittest.TestCase):
    """Test cases for Flask application"""

    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SECRET_KEY'] = 'test-secret-key'
        self.client = self.app.test_client()

    def test_index_page_loads(self):
        """Test that index page loads successfully"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI\xe8\xa1\xa3\xe6\x9c\xab\xe8\xa9\x95\xe4\xbe\xa1\xe5\xae\x9f\xe9\xa8\x93', response.data)  # "AI衣服評価実験" in UTF-8

    def test_allowed_file_function(self):
        """Test allowed_file function"""
        self.assertTrue(allowed_file('image.jpg'))
        self.assertTrue(allowed_file('image.jpeg'))
        self.assertTrue(allowed_file('image.png'))
        self.assertFalse(allowed_file('image.gif'))
        self.assertFalse(allowed_file('image.txt'))
        self.assertFalse(allowed_file('noextension'))

    def test_get_image_media_type(self):
        """Test get_image_media_type function"""
        self.assertEqual(get_image_media_type('image.jpg'), 'image/jpeg')
        self.assertEqual(get_image_media_type('image.jpeg'), 'image/jpeg')
        self.assertEqual(get_image_media_type('image.png'), 'image/png')

    def test_encode_image_to_base64(self):
        """Test image encoding to base64"""
        # Create a test image
        test_image_path = 'test_image.jpg'
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_image_path)

        try:
            # Test encoding
            base64_string = encode_image_to_base64(test_image_path)
            self.assertIsNotNone(base64_string)
            self.assertIsInstance(base64_string, str)
            self.assertTrue(len(base64_string) > 0)
        finally:
            # Clean up
            if os.path.exists(test_image_path):
                os.remove(test_image_path)

    def test_second_page_requires_session(self):
        """Test that second page requires session data"""
        response = self.client.get('/second')
        # Should redirect to index if no session
        self.assertEqual(response.status_code, 302)

    def test_output_page_requires_session(self):
        """Test that output page requires session data"""
        response = self.client.get('/output')
        # Should redirect to index if no session
        self.assertEqual(response.status_code, 302)

    def test_thanks_page_loads(self):
        """Test that thanks page loads"""
        response = self.client.get('/thanks-page')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Thanks', response.data.decode('utf-8', errors='ignore'))

    def test_404_error_handler(self):
        """Test 404 error handler"""
        response = self.client.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)

    @patch('app.extract_criteria_from_images')
    @patch('app.send_to_n8n')
    def test_index_post_with_valid_files(self, mock_send_to_n8n, mock_extract_criteria):
        """Test POST request to index with valid files"""
        # Mock the OpenAI API call
        mock_extract_criteria.return_value = "・シンプルなデザイン\n・明るい色合い"
        mock_send_to_n8n.return_value = True

        # Create test image files
        test_images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), color='red')
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            test_images.append(('like_images', (f'test{i}.jpg', img_bytes, 'image/jpeg')))

        # Send POST request
        response = self.client.post(
            '/',
            data={'account_name': 'test_user'},
            content_type='multipart/form-data'
        )

        # Check response (should redirect to /second)
        self.assertIn(response.status_code, [200, 302])

    def test_file_upload_validation(self):
        """Test file upload validation"""
        # Test with missing account name
        response = self.client.post(
            '/',
            data={},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)


class UtilityFunctionsTestCase(unittest.TestCase):
    """Test utility functions"""

    def test_allowed_file_with_various_extensions(self):
        """Test allowed_file with various extensions"""
        valid_files = ['photo.jpg', 'image.jpeg', 'picture.png']
        invalid_files = ['document.pdf', 'video.mp4', 'text.txt']

        for filename in valid_files:
            self.assertTrue(allowed_file(filename), f"{filename} should be allowed")

        for filename in invalid_files:
            self.assertFalse(allowed_file(filename), f"{filename} should not be allowed")

    def test_get_image_media_type_case_insensitive(self):
        """Test that get_image_media_type is case insensitive"""
        self.assertEqual(get_image_media_type('image.JPG'), 'image/jpeg')
        self.assertEqual(get_image_media_type('image.PNG'), 'image/png')
        self.assertEqual(get_image_media_type('image.Jpg'), 'image/jpeg')


class DirectoryStructureTestCase(unittest.TestCase):
    """Test directory structure"""

    def test_required_directories_exist(self):
        """Test that required directories exist"""
        self.assertTrue(os.path.isdir('templates'), "templates directory should exist")
        self.assertTrue(os.path.isdir('static'), "static directory should exist")
        self.assertTrue(os.path.isdir('test_data'), "test_data directory should exist")

    def test_required_files_exist(self):
        """Test that required files exist"""
        required_files = [
            'app.py',
            'requirements.txt',
            'Procfile',
            'templates/index.html',
            'templates/second.html',
            'templates/output.html',
            'templates/thanks.html',
            'static/style.css',
        ]

        for filepath in required_files:
            self.assertTrue(os.path.isfile(filepath), f"{filepath} should exist")

    def test_test_data_images_exist(self):
        """Test that test data images exist"""
        for i in range(1, 16):
            img_path = f'test_data/img{i}.jpg'
            self.assertTrue(os.path.isfile(img_path), f"{img_path} should exist")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(FlaskAppTestCase))
    suite.addTests(loader.loadTestsFromTestCase(UtilityFunctionsTestCase))
    suite.addTests(loader.loadTestsFromTestCase(DirectoryStructureTestCase))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

