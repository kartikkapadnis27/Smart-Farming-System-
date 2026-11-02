import io
import os
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

from routes.sahayak import sahayak_bp


class SahayakRoutesTestCase(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.config['SECRET_KEY'] = 'test-secret'
        app.register_blueprint(sahayak_bp)
        self.client = app.test_client()

    def test_chat_requires_gemini_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post('/api/chat', json={'input_text': 'Hello'})

        self.assertEqual(response.status_code, 503)
        self.assertIn('GEMINI_API_KEY', response.get_json()['error'])

    def test_chat_uses_gemini_and_farmer_analysis(self):
        model = MagicMock()
        model.generate_content.return_value.text = 'नमस्कार!'
        with self.client.session_transaction() as session:
            session['analysis'] = {'current_profit_total': 100}

        with patch('routes.sahayak._get_gemini_model', return_value=model):
            response = self.client.post('/api/chat', json={'input_text': 'मला मदत हवी'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['text'], 'नमस्कार!')
        self.assertEqual(response.get_json()['show_visual'], 'profit_chart')
        self.assertIn('current_profit_total', model.generate_content.call_args.args[0])

    def test_chat_rejects_empty_message(self):
        response = self.client.post('/api/chat', json={'input_text': '  '})

        self.assertEqual(response.status_code, 400)

    def test_audio_transcription_returns_gemini_transcript_without_saving(self):
        model = MagicMock()
        model.generate_content.return_value.text = 'माझे पीक सोयाबीन आहे'

        with patch('routes.sahayak._get_gemini_model', return_value=model):
            response = self.client.post(
                '/api/audio_upload',
                data={'audio': (io.BytesIO(b'audio-data'), 'recording.webm', 'audio/webm')},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'transcript': 'माझे पीक सोयाबीन आहे'})
        audio_part = model.generate_content.call_args.args[0][1]
        self.assertEqual(audio_part.inline_data.data, b'audio-data')

    def test_audio_upload_rejects_unsupported_format(self):
        response = self.client.post(
            '/api/audio_upload',
            data={'audio': (io.BytesIO(b'data'), 'recording.txt', 'text/plain')},
        )

        self.assertEqual(response.status_code, 415)


if __name__ == '__main__':
    unittest.main()
