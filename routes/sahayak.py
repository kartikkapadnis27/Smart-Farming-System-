import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request, send_from_directory, session
import google.generativeai as genai


sahayak_bp = Blueprint('sahayak', __name__)
logger = logging.getLogger(__name__)
MAX_AUDIO_SIZE = 10 * 1024 * 1024
SUPPORTED_AUDIO_TYPES = {
    'audio/aac',
    'audio/mp4',
    'audio/mpeg',
    'audio/ogg',
    'audio/wav',
    'audio/webm',
    'audio/x-wav',
}


def _get_gemini_model():
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


@sahayak_bp.route('/api/chat', methods=['POST'])
def api_chat():
    """Generate an agriculture-focused response using Gemini."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'A JSON object is required'}), 400

    user_input = data.get('input_text')
    if not isinstance(user_input, str) or not user_input.strip():
        return jsonify({'error': 'Please enter a message'}), 400
    user_input = user_input.strip()
    if len(user_input) > 2000:
        return jsonify({'error': 'Message must be 2000 characters or fewer'}), 400

    model = _get_gemini_model()
    if model is None:
        return jsonify({
            'error': 'AI assistant is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY.'
        }), 503

    analysis = session.get('analysis')
    prompt = (
        'You are Telhan Sathi, a helpful agricultural assistant for Indian farmers. '
        'Reply concisely in the language used by the farmer. Be practical and do not '
        'invent guaranteed prices, yields, or government benefits. Clearly state when '
        'advice depends on local conditions.'
    )
    if analysis and isinstance(analysis, dict):
        prompt += f'\nAvailable farm profitability analysis: {analysis}'
    prompt += f'\nFarmer message: {user_input}'

    try:
        response = model.generate_content(prompt)
        bot_text = response.text
        if not bot_text or not bot_text.strip():
            return jsonify({'error': 'The AI assistant returned an empty response'}), 502
    except Exception:
        logger.exception('Gemini chat generation failed')
        return jsonify({'error': 'The AI assistant could not process your message'}), 502

    return jsonify({
        'text': bot_text.strip(),
        'audio_url': None,
        'show_visual': 'profit_chart' if analysis else None,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@sahayak_bp.route('/api/audio_upload', methods=['POST'])
def api_audio_upload():
    """Transcribe a short audio recording with Gemini without storing the upload."""
    if request.content_length and request.content_length > MAX_AUDIO_SIZE + 1024 * 1024:
        return jsonify({'error': 'Audio must be 10 MB or smaller'}), 413

    audio = request.files.get('audio')
    if not audio or not audio.filename:
        return jsonify({'error': 'Please upload an audio recording'}), 400

    mime_type = (audio.mimetype or '').split(';', 1)[0].strip().lower()
    if mime_type not in SUPPORTED_AUDIO_TYPES:
        return jsonify({'error': 'Unsupported audio format'}), 415

    audio_data = audio.stream.read(MAX_AUDIO_SIZE + 1)
    if not audio_data:
        return jsonify({'error': 'The uploaded audio is empty'}), 400
    if len(audio_data) > MAX_AUDIO_SIZE:
        return jsonify({'error': 'Audio must be 10 MB or smaller'}), 413

    model = _get_gemini_model()
    if model is None:
        return jsonify({
            'error': 'Voice transcription is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY.'
        }), 503

    try:
        audio_part = genai.protos.Part(
            inline_data=genai.protos.Blob(mime_type=mime_type, data=audio_data)
        )
        response = model.generate_content([
            'Transcribe the spoken words exactly in the language spoken. '
            'Return only the transcription, without explanation.',
            audio_part,
        ])
        transcript = response.text
        if not transcript or not transcript.strip():
            return jsonify({'error': 'No speech could be transcribed from this recording'}), 422
    except Exception:
        logger.exception('Gemini audio transcription failed')
        return jsonify({'error': 'The audio could not be transcribed'}), 502

    return jsonify({'transcript': transcript.strip()}), 200


@sahayak_bp.route('/static/audio/<path:filename>')
def serve_audio(filename):
    root = os.path.join(current_app.root_path, 'static', 'audio')
    return send_from_directory(root, filename)
