import os
import threading
import uuid
import logging
from flask import Flask, render_template, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename

from datetime import datetime, timedelta
from config import UPLOAD_FOLDER, OUTPUT_FOLDER, MAX_CONTENT_LENGTH, progress_tracker
from processors import ProofreadingProcessor, TranslationProcessor, OCRProcessor
from document_handler import DocumentHandler


logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def process_document_background(job_id, mode, input_path, language, source_lang, target_lang, original_filename, user_prompt=''):
    """Background processing function"""
    try:
        # Get API keys from environment
        vision_api_key = os.getenv('GOOGLE_VISION_API_KEY', '')
        gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        
        output_filename = None
        
        if mode == 1:
            # OCR Only
            ocr = OCRProcessor(vision_api_key, job_id)
            text = ocr.perform_ocr(input_path)
            output_filename = f"{job_id}_ocr_raw.docx"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            DocumentHandler.save_raw_docx(text, output_path)
            
        elif mode == 2:
            # OCR + Proofread
            ocr = OCRProcessor(vision_api_key, job_id)
            text = ocr.perform_ocr(input_path)
            text = text.replace('\n', '\r')
            proofreader = ProofreadingProcessor(gemini_api_key, job_id=job_id)
            chunks = proofreader.chunk_text(text)
            corrected_chunks = proofreader.process_chunks_parallel(
                chunks,
                lambda chunk: proofreader.proofread_chunk(chunk, language),
                "Proofreading"
            )
            output_filename = f"{job_id}_ocr_proofread.docx"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            DocumentHandler.create_formatted_document(corrected_chunks, output_path, language, "OCR + Proofread")
            
        elif mode == 3:
            # Proofread Only
            content = DocumentHandler.read_docx(input_path)
            
            proofreader = ProofreadingProcessor(gemini_api_key, job_id=job_id)
            chunks = proofreader.chunk_text(content)
            corrected_chunks = proofreader.process_chunks_parallel(
                chunks,
                lambda chunk: proofreader.proofread_chunk(chunk, language),
                "Proofreading"
            )
            
            output_filename = f"{job_id}_proofread.docx"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            DocumentHandler.create_formatted_document(corrected_chunks, output_path, language, "Proofread")
            
        elif mode == 4:
            # OCR + Translation
            ocr = OCRProcessor(vision_api_key, job_id)
            text = ocr.perform_ocr(input_path)
            
            translator = TranslationProcessor(gemini_api_key, job_id=job_id)
            chunks = translator.chunk_text(text)
            translated_chunks = translator.process_chunks_parallel(
                chunks,
                lambda chunk: translator.translate_chunk(chunk, source_lang, target_lang),
                "Translation"
            )
            
            output_filename = f"{job_id}_ocr_translated_{target_lang}.docx"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            DocumentHandler.create_formatted_document(translated_chunks, output_path, target_lang, "OCR + Translated")
            
        elif mode == 5:
            # Translation Only
            content = DocumentHandler.read_docx(input_path)
            
            translator = TranslationProcessor(gemini_api_key, job_id=job_id)
            chunks = translator.chunk_text(content)
            translated_chunks = translator.process_chunks_parallel(
                chunks,
                lambda chunk: translator.translate_chunk(chunk, source_lang, target_lang),
                "Translation"
            )
            
            output_filename = f"{job_id}_translated_{target_lang}.docx"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            DocumentHandler.create_formatted_document(translated_chunks, output_path, target_lang, "Translated")
        
        
        # Mark as complete
        progress_tracker[job_id] = {
            'current': 100,
            'total': 100,
            'status': 'Complete',
            'percentage': 100,
            'output_file': output_filename
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        progress_tracker[job_id] = {
            'current': 0,
            'total': 100,
            'status': f'Error: {str(e)}',
            'percentage': 0,
            'error': True
        }


@app.get("/")
def initialize():
    return render_template('feature.html')

@app.get('/features')
def feature():
    return render_template('feature.html')

@app.get('/login')
def login():
    return render_template('login.html')

@app.get('/register')
def register():
    return render_template('register.html')

@app.get('/pricing')
def pricing():
    return render_template('pricing.html')

@app.get('/contactus')
def contactus():
    return render_template('contactus.html')

@app.get('/upload_document')
def upload_document():
    return render_template('upload_document.html')

@app.get('/tool')
def index_redirect():
    return render_template('index.html')

@app.route('/mode/<int:mode_num>')
def mode_page(mode_num):
    if mode_num not in range(1, 6):
        return "Invalid mode", 404
    return render_template(f'mode{mode_num}.html', mode=mode_num)

@app.route('/process', methods=['POST'])
def process_file():
    try:
        mode = int(request.form.get('mode'))
        file = request.files.get('file')
        language = request.form.get('language')
        source_lang = request.form.get('source_lang')
        target_lang = request.form.get('target_lang')
        user_prompt = request.form.get('user_prompt', '')  # For mode 6
        
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(input_path)
        
        # Initialize progress
        progress_tracker[job_id] = {
            'current': 0,
            'total': 100,
            'status': 'Starting...',
            'percentage': 0
        }
        
        # Process in background thread
        thread = threading.Thread(
            target=process_document_background,
            args=(job_id, mode, input_path, language, source_lang, target_lang, filename, user_prompt)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id})
        
    except Exception as e:
        logger.error(f"Error in process_file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get progress for a job"""
    if job_id in progress_tracker:
        return jsonify(progress_tracker[job_id])
    return jsonify({'error': 'Job not found'}), 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed file"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        return "File not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)