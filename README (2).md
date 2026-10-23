# Smart Multilingual Document Archiving System

Gradio application for uploading, extracting, classifying, searching, and managing legal/policy documents in Arabic, English, and Russian.

## Requirements

- Python 3.10 or 3.11 recommended
- Tesseract OCR installed on the host, with language packs for the languages you need (`ara`, `eng`, `rus`)
- Model artifacts and Excel datasets (not included; see below)

### Install

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Tesseract separately:
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng tesseract-ocr-rus`
- macOS (Homebrew): `brew install tesseract tesseract-lang`
- Windows: install Tesseract OCR and ensure its executable is on PATH.

## Add your private assets

1. Put trained model files in `models/` as documented in `models/README.md`.
2. Put the three language Excel datasets in `data/datasets/` as documented in `data/datasets/README.md`.
3. These files are deliberately excluded from Git. Use a private storage location or a secure download process for deployment; do not publish confidential datasets or credentials.

The application can launch without model files, but classification will return `Models not installed` until they are supplied.

## Configure login

Set manager and/or employee credentials as environment variables before launch:

```bash
# macOS/Linux
export APP_MANAGER_USERNAME='your-manager'
export APP_MANAGER_PASSWORD='use-a-long-secret'
export APP_EMPLOYEE_USERNAME='your-employee'
export APP_EMPLOYEE_PASSWORD='use-a-long-secret'
```

PowerShell:
```powershell
$env:APP_MANAGER_USERNAME="your-manager"
$env:APP_MANAGER_PASSWORD="use-a-long-secret"
```

Only accounts whose username and password environment variables are set are enabled. Never commit `.env` or real credentials.

## Run

```bash
python app.py
```

Gradio prints the local URL in the terminal. For deployment, configure the platform's secrets and persistent storage; ephemeral hosting may lose uploaded documents and generated archives.

## Important notes

- The original notebook depended on Google Drive/Colab paths. This repository version uses local project folders.
- This is a cleaned compatibility pass, not a guarantee of full runtime correctness: test with your actual trained artifacts and sample files before deployment.
- Review authorization and data retention carefully before exposing the app publicly.
