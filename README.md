# Smart Archiving System

Gradio-based multilingual document archiving and classification application (Arabic, English, Russian).

## Project files
- `app.py` — application source (adapted from the supplied notebook export).
- `requirements.txt` — Python dependencies.

## Setup
1. Install Python 3.10+ and Tesseract OCR (including Arabic and English language data).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure dataset, model, output and archive paths for your environment. The supplied source currently contains Google Drive/Colab paths and loads pre-trained model files that are not included in this repository.
4. Run in the environment where those datasets and model artifacts are available:
   ```bash
   python app.py
   ```

## Important before publishing
- Do not commit datasets, trained model files, archives, or personal documents.
- Replace hard-coded demo usernames/passwords with secure authentication; do not publish real credentials.
- The source was extracted from a Markdown export and cleaned for repository organization. It may still require syntax/indentation repair and adaptation from Google Colab to a regular Python environment before it runs.
