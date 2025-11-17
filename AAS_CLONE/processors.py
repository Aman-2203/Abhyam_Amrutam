import time
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import re
import logging

try:
    import fitz  # PyMuPDF
    import requests
    import google.generativeai as genai
    from PIL import Image
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nPlease install required packages:")
    print("pip install PyMuPDF requests google-generativeai python-docx Pillow Flask")
    import sys
    sys.exit(1)

from config import progress_tracker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Base class for document processing operations"""
    
    def __init__(self, gemini_api_key: str, max_workers: int = 5, job_id: str = None):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.max_workers = max_workers
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.2
        self.job_id = job_id
    
    def update_progress(self, current: int, total: int, status: str):
        """Update progress for the job"""
        if self.job_id:
            progress_tracker[self.job_id] = {
                'current': current,
                'total': total,
                'status': status,
                'percentage': int((current / total) * 100) if total > 0 else 0
            }
    
    def chunk_text(self, text: str, max_chunk_size: int = 20000) -> List[str]:
        """Split text into chunks for processing"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                sentences = paragraph.split('।')
                if len(sentences) == 1:
                    sentences = paragraph.split('\t')
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '\t'
                    else:
                        current_chunk += sentence + '\t'
            else:
                if len(current_chunk + paragraph) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '\n\n'
                else:
                    current_chunk += paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
        
    def process_with_rate_limit(self, process_func, *args):
        """Execute function with rate limiting"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        return process_func(*args)
    
    def process_chunks_parallel(self, chunks: List[str], process_func, operation_name: str = "Processing"):
        """Process chunks in parallel with progress tracking"""
        results = [None] * len(chunks)
        total = len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.process_with_rate_limit, process_func, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            completed = 0
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                self.update_progress(completed, total, f"{operation_name}: {completed}/{total}")
                
                try:
                    result = future.result()
                    results[index] = result if result else chunks[index]
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                        logger.warning(f"Rate limit hit on chunk {index + 1}. Retrying...")
                        time.sleep(5)
                        try:
                            result = self.process_with_rate_limit(process_func, chunks[index])
                            results[index] = result if result else chunks[index]
                        except Exception as retry_error:
                            logger.warning(f"Failed chunk {index + 1} after retry: {retry_error}")
                            results[index] = chunks[index]
                    else:
                        logger.warning(f"Error on chunk {index + 1}: {e}")
                        results[index] = chunks[index]
        
        return results


class ProofreadingProcessor(DocumentProcessor):
    """Handles AI-powered proofreading"""
    
    def __init__(self, gemini_api_key: str, max_workers: int = 5, job_id: str = None):
        super().__init__(gemini_api_key, max_workers, job_id)
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def proofread_chunk(self, text_chunk: str, language: str) -> str:
        """Proofread a text chunk"""
        if language.lower() == 'gujarati':
            specific_instructions = """
- Check for proper Gujarati matras (ા, િ, ી, ુ, ૂ, ૃ, ે, ૈ, ો, ૌ)
- Verify correct use of Gujarati conjuncts and half letters
- Check Gujarati punctuation marks (।, ॥, etc.)
- Ensure proper spacing in Gujarati text
- Fix common OCR errors in Gujarati: confusing similar letters (ત/ટ, પ/બ, ક/ખ, etc.)
"""
        else:
            specific_instructions = """
- Check for proper Hindi matras (ा, ि, ी, ु, ू, ृ, े, ै, ो, ौ)
- Verify correct use of Hindi conjuncts and half letters
- Check Hindi punctuation marks (।, ॥, etc.)
- Ensure proper spacing in Hindi text
- Fix common OCR errors in Hindi: confusing similar letters (त/ट, प/फ, क/ख, द/ध, etc.)
"""

        prompt = f"""
ROLE & CORE DIRECTIVE: You are a meticulous digital text restorer. Your sole function is to correct technical errors from an OCR scan while perfectly preserving the original author's voice, style, and intent.

GUIDING PRINCIPLES:
 * The Rule of Minimum Intervention: Only change what is absolutely necessary to fix a clear technical OCR error.
 * The Rule of Stylistic Invisibility: Your corrections must be so perfectly matched to the original style that a reader would never know an OCR error ever existed.

YOUR TASKS (In Order of Priority):
LEVEL 1: PURELY TECHNICAL CORRECTIONS (Mechanical Fixes)
 * Character Recognition: Fix misidentified characters
 * Vowel Marks & Conjuncts ({language}): Correct any missing, extra, or broken matras, bindis/anusvaras, and repair broken conjunct characters
 * Spacing: Eliminate incorrect spaces inside words and add missing spaces between words
 * Punctuation: Correct OCR-mangled punctuation
 * Line Breaks & Hyphenation: Join words incorrectly split by end-of-line hyphenation
 * Formatting & Structure: Reconstruct paragraph breaks, preserve headings
   {specific_instructions}

LEVEL 2: CONTEXT-AWARE CORRECTIONS (Word-Level Fixes)
 * Nonsensical Words: Replace words that are gibberish due to OCR errors
 * Style-Matched Replacement: Replacements MUST match the exact same formality and tone

ABSOLUTE PROHIBITIONS:
  DO NOT TRANSALATE THE CONTENT
 * DO NOT "IMPROVE" THE TEXT
 * DO NOT MODERNIZE OR SANITIZE
 * DO NOT ALTER THE TONE
 * DO NOT CHANGE VOCABULARY LEVEL
 * DO NOT REPHRASE FOR CLARITY

Text to process:
{text_chunk}

Response format:
CORRECTED_TEXT:
[Provide the corrected version with ONLY OCR errors fixed, maintaining the exact original style and tone.]
"""
        
        try:
            response = self.model.generate_content(prompt)
            return self.extract_corrected_text(response.text)
        except Exception as e:
            logger.error(f"Error proofreading chunk: {e}")
            return text_chunk
    
    def extract_corrected_text(self, ai_response: str) -> Optional[str]:
        """Extract corrected text from AI response"""
        try:
            if "CORRECTED_TEXT:" in ai_response:
                parts = ai_response.split("CORRECTED_TEXT:")
                if len(parts) > 1:
                    corrected_part = parts[1]
                    
                    for section in ["CHANGES_MADE:", "FORMATTING_APPLIED:"]:
                        if section in corrected_part:
                            corrected_part = corrected_part.split(section)[0]
                    
                    corrected_text = corrected_part.strip()
                    if corrected_text:
                        return corrected_text
            
            cleaned_response = ai_response.strip()
            prefixes_to_remove = [
                "TECHNICAL ERRORS FOUND:", "CHANGES_MADE:", "FORMATTING_APPLIED:",
                "No technical corrections needed", "No obvious technical errors found"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_response.startswith(prefix):
                    cleaned_response = cleaned_response[len(prefix):].strip()
            
            if len(cleaned_response) > 50:
                return cleaned_response
            
            return None
                    
        except Exception:
            return None


class TranslationProcessor(DocumentProcessor):
    """Handles AI-powered translation"""
    
    def __init__(self, gemini_api_key: str, max_workers: int = 5, job_id: str = None):
        super().__init__(gemini_api_key, max_workers, job_id)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        self.translation_prompt = """You are a master translator and literary stylist specializing in texts with high cultural and religious specificity. Your primary goal is to produce a polished, high-register {target_lang} translation that prioritizes natural flow, contextual dignity, and cultural resonance for the specified audience, moving far beyond literal or word-for-word rendering.

USER QUERY:

Translate the following source document into simple yet professional, highly fluent {target_lang}. CRITICAL OUTPUT RULE: The final response MUST consist SOLELY of the translated text. Do not include any introductory phrases, file headers, AI-generated headings, metadata, commentary, or extraneous text whatsoever.

Preprocessing:
The source text may contain scanning errors (OCR mistakes). Your critical first step is to look past these technical flaws to discern the author's true, intended words and meaning. Do not "correct" the style, only reconstruct the text to make it intelligible for translation.

Translation Strategy & Non-Negotiable Rules:

Step 1: Document Analysis (Internal Only): Before generating the first word of the translation, you must internally analyze the provided text to determine:

Original Writing Style: Identify the core stylistic register (e.g., highly academic, devotional/hymnal, historical narrative, legal/prescriptive, direct instructional, etc.).

Document Type & Genre: Classify the specific type (e.g., philosophical treatise, historical commentary, religious sermon, contemporary report) and the general genre (e.g., philosophy, spirituality, history).

Step 2: Automated Style Directive (Genre-Based Translation): You must automatically select the most appropriate elevated {target_lang} style (e.g., Scholarly/Academic, Devotional/Inspirational, or Modern/Interpretive) that directly corresponds to the identified Document Type & Genre (from Step 1). This ensures the translation's tone and syntax are inherently suited to the text's original purpose, without requiring further user input.

Step 3: Target Audience Adaptation (Indian {target_lang}): The entire tone and lexicon must be optimized for an educated Indian {target_lang}-speaking audience. Favor vocabulary and phrasing that is precise, formal, and widely understood within that context, avoiding overly colloquial, American, or casual Western phrasing.

CRITICAL NOTIFICATION: Jain Terminology Preservation: DO NOT TRANSLATE core Jain religious, philosophical, or technical terms (e.g., Anekantavada, Samyak Charitra, Kevala Jnana, Tirthankara, etc.). These terms must be preserved as they are transliterated in the source text, ensuring the religious and scholarly integrity of the document is maintained. Only surrounding contextual language should be translated. Text within <brackets> should be kept as is since it's usually Sanskrit/technical terms.

Text:
{text_chunk}

Response format:
[Provide the complete translated text maintaining structure and formatting.]
"""
    
    def clean_sanskrit_formatting(self, text: str) -> str:
        """Clean up inconsistent Sanskrit formatting markers"""
        patterns = [
            r'\*sanskrit\*(.*?)\*/sanskrit\*',
            r'\*\*sanskrit\*\*(.*?)\*\*/sanskrit\*\*',
            r'\[sanskrit\](.*?)\[/sanskrit\]',
            r'<sanskrit>(.*?)</sanskrit>',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, r'<\1>', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text
    
    def translate_chunk(self, text_chunk: str, source_lang: str, target_lang: str) -> str:
        """Translate a text chunk"""
        try:
            cleaned_text = self.clean_sanskrit_formatting(text_chunk)
            
            prompt = self.translation_prompt.format(
                target_lang=target_lang,
                text_chunk=cleaned_text
            )
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error translating chunk: {e}")
            return text_chunk


class OCRProcessor:
    """Handles OCR operations"""
    
    def __init__(self, vision_api_key: str, job_id: str = None):
        self.vision_api_key = vision_api_key
        self.vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={vision_api_key}"
        self.job_id = job_id
    
    def update_progress(self, current: int, total: int, status: str):
        """Update progress for the job"""
        if self.job_id:
            progress_tracker[self.job_id] = {
                'current': current,
                'total': total,
                'status': status,
                'percentage': int((current / total) * 100) if total > 0 else 0
            }
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF pages to images"""
        pdf_document = fitz.open(pdf_path)
        images = []
        
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        total_pages = pdf_document.page_count
        
        for page_num in range(total_pages):
            self.update_progress(page_num + 1, total_pages, f"Converting page {page_num + 1}/{total_pages}")
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=matrix)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
        
        pdf_document.close()
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert image to base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def extract_text_from_image(self, base64_image: str) -> str:
        """Extract text using Google Vision API"""
        request_body = {
            "requests": [{
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1}]
            }]
        }
        
        response = requests.post(
            self.vision_api_url,
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=90
        )
        
        if not response.ok:
            error_msg = response.json().get('error', {}).get('message', 'API request failed')
            raise Exception(f"Google Vision API error: {error_msg}")
        
        data = response.json()
        annotations = data.get('responses', [{}])[0].get('textAnnotations', [])
        return annotations[0]['description'] if annotations else ""
    
    def perform_ocr(self, pdf_path: str) -> str:
        """Perform OCR on PDF with parallel processing per page"""
        images = self.pdf_to_images(pdf_path)
        total = len(images)
        extracted_texts = [None] * total

        def ocr_page(index_image_tuple):
            index, image = index_image_tuple
            try:
                self.update_progress(index + 1, total, f"Extracting text from page {index + 1}/{total}")
                base64_image = self.image_to_base64(image)
                text = self.extract_text_from_image(base64_image)
                return (index, text if text.strip() else "")
            except Exception as e:
                logger.warning(f"Error on page {index + 1}: {e}")
                return (index, "")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(ocr_page, (i, img)) for i, img in enumerate(images)]
            for future in as_completed(futures):
                index, text = future.result()
                extracted_texts[index] = text

        return '\n\n'.join([t for t in extracted_texts if t])