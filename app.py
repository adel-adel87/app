# Install required libraries
# Import libraries

import gradio as gr

import os

import datetime

import pandas as pd

import numpy as np

from langdetect import detect, DetectorFactory

import uuid

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

import pdfplumber

import docx2txt

import pytesseract

import cv2

from PIL import Image

import joblib

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification,pipeline

import torch

import tempfile

from io import BytesIO



import shutil

# Ensure consistent language detection

DetectorFactory.seed = 0



# Mount Google Drive

# Define paths

output_dir = str(RESULTS_DIR)

archive_dir = str(ARCHIVE_DIR)

dataset_paths = {

    'ar': str(DATA_DIR / 'readme_ar.xlsx'),

    'en': str(DATA_DIR / 'readme_en.xlsx'),

    'ru': str(DATA_DIR / 'readme_ru.xlsx')

}

os.makedirs(output_dir, exist_ok=True)

os.makedirs(archive_dir, exist_ok=True)



# Load saved models and utilities

print("Loading saved models and utilities...")
svm = rf = vectorizer = le = model_distilbert = tokenizer_distilbert = None
try:
    svm = joblib.load(MODELS_DIR / "svm_model.joblib")
    rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    le = joblib.load(MODELS_DIR / "le.joblib")
    model_distilbert = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR / "distilbert_model")
    tokenizer_distilbert = AutoTokenizer.from_pretrained(MODELS_DIR / "distilberttokenizer_model")
    print("Models loaded.")
except Exception as exc:
    print(f"Model files unavailable; classification is disabled until models are installed: {exc}")

# Define inverse label mapping for subdomains

inverse_label_mapping = ({i: label for i, label in enumerate(le.classes_)} if le is not None else {})



# Define categories

CATEGORIES = ['Legal', 'Policies']  # Domains

SUB_CATEGORIES = ['Constitutions', 'UN Parliament', 'Contracts', 'Olympic Rules', 'Code of Conduct', 'Judicial Rulings']  # Sub-domains



print("Models and utilities loaded successfully.")



document_store = []

accounts = {}
for role, env_prefix in (("manager", "APP_MANAGER"), ("employee", "APP_EMPLOYEE")):
    username = os.getenv(f"{env_prefix}_USERNAME")
    password = os.getenv(f"{env_prefix}_PASSWORD")
    if username and password:
        accounts[role] = {"username": username, "password": password, "role": role}

current_user = {"username": None, "role": None}

# Helper function to update categories

def update_categories():

    global SUB_CATEGORIES

    try:

        categories = set()

        for lang, path in dataset_paths.items():

            if os.path.exists(path):

                df = pd.read_excel(path)

                if 'Sub-domain' in df.columns:

                    categories.update(df['Sub-domain'].dropna().unique())

        if categories:

            SUB_CATEGORIES = sorted(list(categories))

            return f"Categories updated: {', '.join(SUB_CATEGORIES)}"

        else:

            return "No categories found in datasets. Using default categories."

    except Exception as e:

        print(f"Error updating categories: {e}")

        return f"Error updating categories: {e}. Using default categories."



# Helper function for browsing

def get_subcategories():

    print("Running get_subcategories()...")

    try:

        if not os.path.exists(output_dir):

            print(f"Error: Directory '{output_dir}' does not exist.")

            return []



        subcategories = []

        for name in os.listdir(output_dir):

            full_path = os.path.join(output_dir, name)

            if os.path.isdir(full_path):

                subcategories.append(name)

            else:

                print(f"Skipping '{name}' as it is not a directory.")



        if not subcategories:

            print("No subcategories found in results folder.")

            return []



        subcategories = sorted(subcategories)

        print(f"Found subcategories: {subcategories}")

        return subcategories

    except Exception as e:

        print(f"Error loading subcategories: {str(e)}")

        return []



def get_files_in_subfolder(subcategory):

    try:

        if not subcategory or subcategory in ["No subcategories found", "Error loading subcategories"]:

            return []

        folder_path = os.path.join(output_dir, subcategory)

        if not os.path.exists(folder_path):

            return []

        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'))]

        return sorted(files) if files else ["No files found"]

    except Exception as e:

        print(f"Error loading files for {subcategory}: {e}")

        return ["Error loading files"]



# Text Extraction Functions

def extract_text_from_pdf(path):

    try:

        text = ""

        with pdfplumber.open(path) as pdf:

            for page in pdf.pages:

                text += page.extract_text() or "" + "\n"

            pdf.close()

        return text.strip()

    except Exception as e:

        print(f"PDF extraction error: {e}")

        return ""



def extract_text_from_word(path):

    try:

        return docx2txt.process(path).strip()

    except Exception as e:

        print(f"Word extraction error: {e}")

        return ""



def extract_text_from_image(file_path):

    try:

        img = Image.open(file_path).convert('L')

        img = np.array(img)

        img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)

        custom_config = r'--oem 3 --psm 6 -l eng+ara'

        return pytesseract.image_to_string(img, config=custom_config).strip()

    except Exception as e:

        print(f"OCR error: {e}")

        return ""



def read_file_content(file_obj):

    try:

        print(f"File object type: {type(file_obj)}, Name: {file_obj.name}, Attributes: {dir(file_obj)}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as temp_file:

            temp_path = temp_file.name

            if isinstance(file_obj, bytes):

                temp_file.write(file_obj)

            elif isinstance(file_obj, str):

                temp_file.write(file_obj.encode('utf-8'))

            else:

                file_obj.seek(0)

                temp_file.write(file_obj.read())



        content = ""

        if file_obj.name.endswith(".txt"):

            with open(temp_path, 'r', encoding='utf-8') as f:

                content = f.read()

        elif file_obj.name.endswith(".pdf"):

            content = extract_text_from_pdf(temp_path)

        elif file_obj.name.endswith(".docx"):

            content = extract_text_from_word(temp_path)

        elif file_obj.name.lower().endswith((".png", ".jpg", ".jpeg")):

            content = extract_text_from_image(temp_path)

        else:

            return "Unsupported file format.", None



        os.unlink(temp_path)

        if not content or content.isspace():

            return "No readable content extracted from the file.", None

        return None, content

    except Exception as e:

        if 'temp_path' in locals() and os.path.exists(temp_path):

            os.unlink(temp_path)

        return f"Error reading file: {e}", None



def classify_text(text, confidence_threshold=0.6):

    try:

        inputs = tokenizer_distilbert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():

            outputs = model_distilbert(**inputs)

        logits = outputs.logits

        distilbert_proba = torch.softmax(logits, dim=1).numpy()[0]

        distilbert_confidence = np.max(distilbert_proba)

        distilbert_subdomain_encoded = np.argmax(distilbert_proba)

        distilbert_subdomain = inverse_label_mapping[distilbert_subdomain_encoded]



        text_vec = vectorizer.transform([text])

        svm_proba = svm.predict_proba(text_vec)[0]

        svm_confidence = np.max(svm_proba)

        svm_subdomain_encoded = np.argmax(svm_proba)

        svm_subdomain = inverse_label_mapping[svm_subdomain_encoded]



        rf_proba = rf.predict_proba(text_vec)[0]

        rf_confidence = np.max(rf_proba)

        rf_subdomain_encoded = np.argmax(rf_proba)

        rf_subdomain = inverse_label_mapping[rf_subdomain_encoded]



        confidences = [distilbert_confidence, svm_confidence, rf_confidence]

        subdomains = [distilbert_subdomain, svm_subdomain, rf_subdomain]

        max_confidence_idx = np.argmax(confidences)

        final_subdomain = subdomains[max_confidence_idx]

        final_confidence = confidences[max_confidence_idx]



        return final_subdomain, final_confidence

    except Exception as e:

        print(f"Classification error: {e}")

        return "Unknown", 0.0



def extract_keywords(text, top_k=5):

    try:

        vec = TfidfVectorizer(stop_words='english', max_features=top_k)

        tfidf = vec.fit_transform([text])

        keywords = list(vec.get_feature_names_out())

        return keywords if keywords else ["no_keywords"]

    except Exception as e:

        print(f"Keyword extraction error: {e}")

        return ["no_keywords"]



def login(username, password):

    for account in accounts.values():

        if account["username"] == username and account["password"] == password:

            current_user["username"] = username

            current_user["role"] = account["role"]

            return f"Logged in as **{account['role'].capitalize()}**"

    return "**Incorrect** username or password."



def get_file_path(doc_id, language):

    try:

        df = pd.read_excel(dataset_paths[language])

        if 'doc_id' not in df.columns or 'Sub-domain' not in df.columns or 'Filename' not in df.columns:

            print(f"Dataset '{language}' missing required columns for get_file_path.")

            return None

        df['doc_id'] = df['doc_id'].astype(str)

        record = df[df['doc_id'] == str(doc_id)]

        if record.empty:

            print(f"No record found for doc_id '{doc_id}' in {language} dataset.")

            return None

        category = record['Sub-domain'].iloc[0]

        filename_in_dataset = record['Filename'].iloc[0]

        category_folder = os.path.join(output_dir, category)

        file_path = os.path.join(category_folder, filename_in_dataset)

        return file_path if os.path.exists(file_path) else None

    except Exception as e:

        print(f"Error finding file path for doc_id {doc_id} in {language}: {e}")

        return None



def delete_document(subcategory, filename):

    print(f"delete_document: subcategory={subcategory}, filename={filename}")

    if not current_user["username"]:

        return "Please **log in** first.", None

    if current_user["role"] != "manager":

        return "**Access denied.** Only managers can delete documents.", None

    if not subcategory or not filename or filename == "No files found":

        return "Please select a valid subcategory and file.", None

    try:

        # Delete the file from the output directory

        file_path = os.path.join(output_dir, subcategory, filename)

        deleted_file_path = None

        if os.path.exists(file_path):

            os.remove(file_path)

            deleted_file_path = file_path

            print(f"Deleted file: {file_path}")

        else:

            print(f"File not found at: {file_path}")



        # Remove dataset entry from all languages

        deleted_from_dataset = False

        for lang, dataset_path in dataset_paths.items():

            if not os.path.exists(dataset_path):

                print(f"Dataset '{lang}' not found at {dataset_path}")

                continue

            try:

                df = pd.read_excel(dataset_path)

                if 'Filename' not in df.columns or 'Sub-domain' not in df.columns:

                    print(f"Dataset '{lang}' missing required columns.")

                    continue

                original_len = len(df)

                df = df[~((df['Filename'] == filename) & (df['Sub-domain'] == subcategory))]

                if len(df) < original_len:

                    cols_to_keep = ['doc_id', 'Domain', 'Sub-domain', 'Paragraph', 'Filename']

                    df = df[cols_to_keep]

                    df.to_excel(dataset_path, index=False)

                    print(f"Removed {original_len - len(df)} entry from {lang} dataset.")

                    deleted_from_dataset = True

            except Exception as e:

                print(f"Error updating {lang} dataset: {e}")



        if deleted_from_dataset:

            update_archive()



        # Update the dropdown with remaining files

        updated_dropdown = update_document_dropdown(subcategory)

        print(f"Updated dropdown after deletion: {updated_dropdown.choices}")



        if deleted_file_path:

            return f"Document '{filename}' deleted successfully from '{subcategory}' folder and dataset.", updated_dropdown

        return f"Document '{filename}' deleted from '{subcategory}' folder (no dataset entry found).", updated_dropdown

    except Exception as e:

        print(f"Delete error: {e}")

        return f"Error deleting document: {e}", None



def download_document(subcategory, filename):

    print(f"download_document: subcategory={subcategory}, filename={filename}")

    if not current_user["username"]:

        return "Please **log in** first."

    if not subcategory or not filename or filename == "No files found":

        return "Please select a valid subcategory and file."

    try:

        file_path = os.path.join(output_dir, subcategory, filename)

        print(f"Download file path: {file_path}")

        if not os.path.exists(file_path):

            return f"File '{filename}' not found in '{subcategory}' folder."

        return file_path

    except Exception as e:

        print(f"Download error: {e}")

        return f"Error downloading document: {e}"



def update_archive():

    archive_data = []

    for lang, path in dataset_paths.items():

        try:

            df = pd.read_excel(path)

            df['language'] = lang

            archive_data.append(df)

        except Exception as e:

            print(f"Error reading {lang} dataset for archive: {e}")

    if archive_data:

        combined_archive = pd.concat(archive_data, ignore_index=True)

        # Ensure 'Filename' is included if it exists in any of the dataframes

        cols_to_keep = ['doc_id', 'Domain', 'Sub-domain', 'Paragraph', 'language']

        if 'Filename' in combined_archive.columns:

             cols_to_keep.append('Filename')

        combined_archive = combined_archive.get(cols_to_keep, combined_archive) # Use .get to avoid KeyError if 'Filename' is not in combined_archive

        archive_path_csv = os.path.join(archive_dir, 'archive_data.csv')

        archive_path_excel = os.path.join(archive_dir, 'archive_data.xlsx')

        combined_archive.to_csv(archive_path_csv, index=False)

        combined_archive.to_excel(archive_path_excel, index=False)

        print("Archive updated.")



def browse_by_category(subcategory, filename):

    if not current_user["username"]:

        return "Please **log in** first.", "" # Return empty doc_id

    if not subcategory or subcategory in ["No subcategories found", "Error loading subcategories"]:

        return "Please select a valid subcategory.", "" # Return empty doc_id

    if not filename or filename == "No files found":

        return "Please select a valid file.", "" # Return empty doc_id

    try:

        file_path = os.path.join(output_dir, subcategory, filename)

        if not os.path.exists(file_path):

            return f"File '{filename}' not found in '{subcategory}' folder.", "" # Return empty doc_id



        content = ""

        if filename.endswith(".txt"):

            with open(file_path, 'r', encoding='utf-8') as f:

                content = f.read()

        elif filename.endswith(".pdf"):

            content = extract_text_from_pdf(file_path)

        elif filename.endswith(".docx"):

            content = extract_text_from_word(file_path)

        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):

            content = extract_text_from_image(file_path)

        else:

            return "Unsupported file format.", "" # Return empty doc_id



        if not content or content.isspace():

            return "No readable content extracted from the file.", "" # Return empty doc_id



        # Find the doc_id associated with this filename in the datasets

        doc_id = "Not Found"

        for lang, path in dataset_paths.items():

            try:

                df = pd.read_excel(path)

                # Now we can reliably use the 'Filename' column to find the doc_id

                if 'Filename' in df.columns and 'doc_id' in df.columns:

                    # Escape special characters in filename for regex matching if needed,

                    # but direct string comparison might be sufficient if filenames are exact.

                    # Let's use direct string comparison for now.

                    matching_rows = df[df['Filename'] == filename]

                    if not matching_rows.empty:

                        doc_id = matching_rows['doc_id'].iloc[0]

                        break # Found the doc_id in this language dataset

            except Exception as e:

                print(f"Error searching dataset {lang} for filename {filename}: {e}")



        return f"### Content of {filename}\n{content}", f"{doc_id}" # Return content and doc_id

    except Exception as e:

        return f"Error reading file: {e}", "" # Return error message and empty doc_id



def get_doc_ids_by_language(language):

    if not current_user["username"]:

        return []

    if language not in dataset_paths:

        return []

    try:

        df = pd.read_excel(dataset_paths[language])

        if 'doc_id' not in df.columns:

            return []

        # Ensure doc_id is string for comparison

        return df['doc_id'].astype(str).tolist()

    except Exception as e:

        print(f"Error loading dataset for language {language}: {e}")

        return []



def update_doc_ids(language):

    return gr.Dropdown(choices=get_doc_ids_by_language(language), interactive=True)





def view_document_by_id(language, doc_id):

    if not current_user["username"]:

        return "Please **log in** first."

    if language not in dataset_paths:

        return f"Invalid language selected: {language}"

    try:

        df = pd.read_excel(dataset_paths[language])

        if 'doc_id' not in df.columns or 'Paragraph' not in df.columns:

            return "Dataset does not contain required columns: 'doc_id' and 'Paragraph'."

        df['doc_id'] = df['doc_id'].astype(str)

        result = df[df['doc_id'] == str(doc_id)]['Paragraph']

        if result.empty:

            return f"No document found with doc_id '{doc_id}' in {language} dataset."

        return result.iloc[0]

    except Exception as e:

        return f"Error retrieving document: {e}"



def get_abstract_from_file(subcategory, filename):

    print(f"get_abstract_from_file: subcategory={subcategory}, filename={filename}")

    if not current_user["username"]:

        return "Please **log in** first."

    if not subcategory or not filename or filename == "No files found":

        return "Please select a valid subcategory and file."

    try:

        file_path = os.path.join(output_dir, subcategory, filename)

        print(f"Abstract file path: {file_path}")

        if not os.path.exists(file_path):

            return f"File '{filename}' not found in '{subcategory}' folder."



        content = ""

        if filename.endswith(".txt"):

            with open(file_path, 'r', encoding='utf-8') as f:

                content = f.read()

        elif filename.endswith(".pdf"):

            content = extract_text_from_pdf(file_path)

        elif filename.endswith(".docx"):

            content = extract_text_from_word(file_path)

        elif filename.lower().endswith((".png", ".jpg", ".jpeg")):

            content = extract_text_from_image(file_path)

        else:

            return "Unsupported file format."



        if not content or content.isspace():

            return "No readable content extracted from the file."



        # Find the first sentence (up to the first period)

        first_period = content.find('.')

        if first_period == -1:

            words = content.split()

            return f"Abstract: {' '.join(words[:100])}{'...' if len(words) > 100 else ''}"



        return f"Abstract: {content[:first_period + 1].strip()}"

    except Exception as e:

        print(f"Abstract error: {e}")

        return f"Error retrieving abstract: {e}"



def show_index():

    if not current_user["username"]:

        return "Please **log in** first."

    output = []

    for lang, path in dataset_paths.items():

        try:

            df = pd.read_excel(path)

            if not all(col in df.columns for col in ['doc_id', 'Domain', 'Sub-domain', 'Paragraph']):

                output.append(f"Error: {lang} dataset missing required columns.")

                continue

            df['doc_id'] = df['doc_id'].astype(str)

            output.append(f"### {lang.upper()} Dataset")

            for _, row in df.iterrows():

                output.append(f"- **Doc ID**: {row['doc_id']}")

                output.append(f"  **Domain**: {row['Domain']}")

                output.append(f"  **Sub-domain**: {row['Sub-domain']}")

                output.append(f"  **Paragraph**: {row['Paragraph'][:200]}..." if len(row['Paragraph']) > 200 else row['Paragraph'])

                output.append("")

            output.append("")

        except Exception as e:

            output.append(f"Error loading {lang} dataset: {e}")

    return "\n".join(output) if output else "**No documents** available in any dataset."



def show_classifications():

    if not current_user["username"]:

        return "Please **log in** first."

    if not document_store:

        return "**No classification data**."

    return "\n".join([f"- **{doc['title']}** | Category: {doc['category']} | Accuracy: {round(doc['confidence'] * 100, 2)}%" for doc in document_store])



def search_documents(language, search_term):

    if not current_user["username"]:

        return "Please **log in** first."

    if not search_term or search_term.isspace():

        return "Please enter a valid search term."

    if language not in dataset_paths:

        return f"Invalid language selected: {language}"

    try:

        df = pd.read_excel(dataset_paths[language])

        if 'doc_id' not in df.columns or 'Paragraph' not in df.columns:

            return f"{language} dataset does not contain required columns: 'doc_id' and 'Paragraph'."

        df['doc_id'] = df['doc_id'].astype(str)

        results = df[df['Paragraph'].str.contains(search_term, case=False, na=False)][['doc_id', 'Paragraph']]

        if results.empty:

            return f"No documents found containing the term '{search_term}' in {language} dataset."

        output = [f"### Search Results ({language})"]

        for _, row in results.iterrows():

            output.append(f"- **Doc ID**: {row['doc_id']}")

            output.append(f"  **Paragraph**: {row['Paragraph'][:200]}..." if len(row['Paragraph']) > 200 else row['Paragraph'])

            output.append("")

        return "\n".join(output)

    except Exception as e:

        return f"Error searching documents: {e}"



def upload_and_analyze(file, title):

    global document_store

    if not current_user["username"]:

        return "Please **log in** first."



    file_obj = None

    content = None

    err = None



    try:

        if not title or title.isspace():

             return "**Error:** Please provide a document title."



        if isinstance(file, str):

            file_path = file

            if file_path.endswith(".txt"):

                with open(file_path, 'r', encoding='utf-8') as f:

                    content = f.read()

                err = None

            elif file_path.endswith(".pdf"):

                content = extract_text_from_pdf(file_path)

                err = None

            elif file_path.endswith(".docx"):

                content = extract_text_from_word(file_path)

                err = None

            elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):

                content = extract_text_from_image(file_path)

                err = None

            else:

                err = "**Unsupported** file format."

            file_obj = open(file_path, 'rb') if file_path and os.path.exists(file_path) else None

        else:

            file_obj = file

            err, content = read_file_content(file_obj)



        if err:

            return err



        if not content or content.isspace():

            return "**No readable content** extracted from the file."



        category, confidence = classify_text(content)

        domain = 'Legal' if category in ['Constitutions', 'Contracts', 'Judicial Rulings'] else 'Policies'

        keywords = extract_keywords(content)

        file_type = (
            "PDF" if getattr(file_obj, "name", "").lower().endswith(".pdf") else
            "DOCX" if getattr(file_obj, "name", "").lower().endswith(".docx") else
            "IMAGE" if getattr(file_obj, "name", "").lower().endswith((".png", ".jpg", ".jpeg")) else
            "TXT" if getattr(file_obj, "name", "").lower().endswith(".txt") else "Unknown"
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



        try:

            language = detect(content)

            if language not in ['ar', 'en', 'ru']:

                return f"Detected language '{language}' not supported. Only 'ar', 'en', 'ru' are supported."

        except Exception as e:

            return f"Language detection error: {e}"



        doc_id = str(uuid.uuid4())

        # Use only the sanitized title for the filename, retaining original extension

        original_filename, file_extension = os.path.splitext(file.name if hasattr(file, 'name') else title)

        sanitized_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in title).strip() # Allow spaces, underscores, hyphens

        if not sanitized_title:

             sanitized_title = "untitled_document"

        final_filename = f"{sanitized_title}{file_extension}"



        folder_name = category

        folder_path = os.path.join(output_dir, folder_name)

        os.makedirs(folder_path, exist_ok=True)



        save_path = os.path.join(folder_path, final_filename)



        # Check if a file with the same name already exists in the category folder

        if os.path.exists(save_path):

            base, ext = os.path.splitext(final_filename)

            count = 1

            while os.path.exists(f"{base}_{count}{ext}"):

                count += 1

            final_filename = f"{base}_{count}{ext}"

            save_path = os.path.join(folder_path, final_filename)





        if file_obj and hasattr(file_obj, 'read'):

            file_obj.seek(0)

            with open(save_path, "wb") as f:

                f.write(file_obj.read())

        elif isinstance(file, str) and os.path.exists(file):

            shutil.copy(file, save_path)



        document_store.append({

            'title': title,

            'content': content,

            'category': category,

            'file_type': file_type,

            'timestamp': timestamp,

            'confidence': confidence

        })



        try:

            df = pd.read_excel(dataset_paths[language]) if os.path.exists(dataset_paths[language]) else pd.DataFrame(columns=['doc_id', 'Domain', 'Sub-domain', 'Paragraph', 'Filename']) # Added 'Filename' column

            new_row = {

                'doc_id': doc_id,

                'Domain': domain,

                'Sub-domain': category,

                'Paragraph': content,

                'Filename': final_filename # Store the generated filename

            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Ensure columns are in correct order and 'Filename' is included

            cols_order = ['doc_id', 'Domain', 'Sub-domain', 'Paragraph', 'Filename']

            df = df.reindex(columns=cols_order)

            df.to_excel(dataset_paths[language], index=False)

        except Exception as e:

            return f"Error updating {language} dataset: {e}"



        update_archive()



        result = f"""

        ### Analysis Result

        **Document ID:** {doc_id}

        **Title:** {title}

        **Category (Sub-domain):** {category}

        **Domain:** {domain}

        **Language:** {language}

        **Keywords:** {', '.join(keywords)}

        **File Type:** {file_type}

        **Classification Accuracy:** {round(confidence * 100, 2)}%

        **Uploaded At:** {timestamp}

        **Saved Filename:** {final_filename}

        """

        return result

    except Exception as e:

        return f"**Error** during analysis: {e}"



with gr.Blocks(title="Smart Archiving System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("<h1 style='text-align: center;'>Smart Archiving System</h1>")

    with gr.Column(scale=0.5, min_width=300):

        gr.Markdown("<h3 style='color: #555;'>User Login</h3>")

        username_input = gr.Textbox(label="Username", placeholder="Enter your username")

        password_input = gr.Textbox(label="Password", type="password", placeholder="Enter your password")

        login_btn = gr.Button("Login", variant="primary")

        login_output = gr.Textbox(label="Login Status", interactive=False)

        login_btn.click(login, inputs=[username_input, password_input], outputs=login_output)



    with gr.Tab("Upload & Analyze"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Upload and Analyze Document</h3>")

            with gr.Row():

                load_categories_btn = gr.Button("Load Categories", variant="secondary")

                category_output = gr.Textbox(label="Category Status", interactive=False)

                load_categories_btn.click(update_categories, outputs=category_output)

            file_input = gr.File(label="Choose File", file_types=['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'])

            title_input = gr.Textbox(label="Document Title", placeholder="Enter document title")

            upload_button = gr.Button("Upload and Analyze", variant="primary")

            result_output = gr.Textbox(label="Analysis Result", interactive=False)

            upload_button.click(upload_and_analyze, inputs=[file_input, title_input], outputs=result_output)



    with gr.Tab("Document Management"):

      with gr.Column():

          gr.Markdown("<h3 style='color: #555;'>Manage Documents</h3>")

          with gr.Row():

              abstract_subcategory_input = gr.Dropdown(choices=get_subcategories(), label="Select Subcategory", interactive=True, value=None)

              abstract_filename_input = gr.Dropdown(choices=[], label="Select Document", interactive=True, value=None)



              def update_document_dropdown(subcategory):

                  print(f"Updating document dropdown for subcategory: {subcategory}")

                  if not subcategory:

                      print("No subcategory selected, returning empty choices.")

                      return gr.Dropdown(choices=[], value=None)

                  files = get_files_in_subfolder(subcategory)

                  print(f"Files returned by get_files_in_subfolder: {files}")

                  valid_files = [f for f in files if isinstance(f, str) and f not in ["No files found", "Error loading files"] and not f.startswith('.')]

                  if not valid_files:

                      print("No valid files found, returning empty choices.")

                      return gr.Dropdown(choices=[], value=None)

                  print(f"Valid files: {valid_files}")

                  return gr.Dropdown(choices=sorted(valid_files), value=None)



              abstract_subcategory_input.change(

                  fn=update_document_dropdown,

                  inputs=abstract_subcategory_input,

                  outputs=abstract_filename_input

              )



              abstract_output = gr.Textbox(label="Document Abstract", interactive=False)

              abstract_button = gr.Button("Get Abstract")

              abstract_button.click(get_abstract_from_file, inputs=[abstract_subcategory_input, abstract_filename_input], outputs=[abstract_output])



          with gr.Row():

              delete_subcategory_input = gr.Dropdown(choices=get_subcategories(), label="Select Subcategory", interactive=True, value=None)

              delete_filename_input = gr.Dropdown(choices=["No subcategory selected"], label="Select Document", interactive=True, value=None)

              delete_subcategory_input.change(

                  fn=update_document_dropdown,

                  inputs=delete_subcategory_input,

                  outputs=delete_filename_input

              )

              def show_document_info(subcategory, filename):

                  print(f"show_document_info: subcategory={subcategory}, filename={filename}")

                  if not current_user["username"]:

                      return "Please **log in** first."

                  if not subcategory or not filename or filename == "No files found" or filename == "No subcategory selected":

                      return "Please select a valid subcategory and file."

                  try:

                      file_path = os.path.join(output_dir, subcategory, filename)

                      print(f"Preview file path: {file_path}")

                      if not os.path.exists(file_path):

                          return f"File '{filename}' not found in '{subcategory}' folder."

                      content = ""

                      if filename.endswith(".txt"):

                          with open(file_path, 'r', encoding='utf-8') as f:

                              content = f.read()

                      elif filename.endswith(".pdf"):

                          content = extract_text_from_pdf(file_path)

                      elif filename.endswith(".docx"):

                          content = extract_text_from_word(file_path)

                      elif filename.lower().endswith((".png", ".jpg", ".jpeg")):

                          content = extract_text_from_image(file_path)

                      else:

                          return "Unsupported file format."

                      if not content or content.isspace():

                          return "No readable content extracted from the file."

                      return f"Preview: {content[:200]}{'...' if len(content) > 200 else ''}"

                  except Exception as e:

                      print(f"Preview error: {e}")

                      return f"Error retrieving document info: {e}"

              delete_info_output = gr.Textbox(label="Selected Document Info (Content Preview)", interactive=False)

              delete_filename_input.change(

                  fn=show_document_info,

                  inputs=[delete_subcategory_input, delete_filename_input],

                  outputs=delete_info_output

              )

          delete_button = gr.Button("Delete Document (Manager Only)", variant="stop")

          delete_output = gr.Textbox(label="Deletion Status", interactive=False)

          delete_button.click(

              fn=delete_document,

              inputs=[delete_subcategory_input, delete_filename_input],

              outputs=[delete_output, delete_filename_input]

          )



          with gr.Row():

              download_subcategory_input = gr.Dropdown(choices=get_subcategories(), label="Select Subcategory", interactive=True, value=None)

              download_filename_input = gr.Dropdown(choices=[], label="Select Document", interactive=True, value=None)

              download_subcategory_input.change(

                  fn=update_document_dropdown,

                  inputs=download_subcategory_input,

                  outputs=download_filename_input

              )

              download_button = gr.Button("Download Document", variant="secondary")

              download_output = gr.File(label="Download File", visible=True)

              download_button.click(

                  fn=download_document,

                  inputs=[download_subcategory_input, download_filename_input],

                  outputs=download_output

              )



    with gr.Tab("Browse by Category"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Browse by Category</h3>")

            with gr.Row():

                subcategory_dropdown = gr.Dropdown(label="Select Subcategory", choices=get_subcategories(), interactive=True, value=None)

                document_dropdown = gr.Dropdown(label="Select Document", choices=[], interactive=True, value=None)

                def update_document_dropdown(subcategory):

                    print(f"Updating document dropdown for subcategory: {subcategory}")

                    if not subcategory:

                        print("No subcategory selected, returning empty choices.")

                        return gr.Dropdown(choices=[], value=None)

                    files = get_files_in_subfolder(subcategory)

                    print(f"Files returned by get_files_in_subfolder: {files}")

                    # Filter out invalid values and handle error cases

                    valid_files = [f for f in files if isinstance(f, str) and f not in ["No files found", "Error loading files"] and not f.startswith('.')]

                    if not valid_files:

                        print("No valid files found, returning empty choices.")

                        return gr.Dropdown(choices=[], value=None)

                    return gr.Dropdown(choices=sorted(valid_files), value=None)

                subcategory_dropdown.change(

                    fn=update_document_dropdown,

                    inputs=subcategory_dropdown,

                    outputs=document_dropdown

                )

            # Add a textbox to display the doc_id

            doc_id_display = gr.Textbox(label="Document ID", interactive=False)

            content_output = gr.Textbox(label="Document Content", interactive=False, lines=20)

            document_dropdown.change(

                fn=browse_by_category,

                inputs=[subcategory_dropdown, document_dropdown],

                outputs=[content_output, doc_id_display] # Output both content and doc_id

            )



    with gr.Tab("View Document by ID"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>View Document by ID</h3>")

            language_input = gr.Dropdown(choices=['ar', 'en', 'ru'], label="Select Language")

            doc_id_input = gr.Dropdown(choices=[], label="Select Document ID", interactive=True)

            view_button = gr.Button("View Document", variant="secondary")

            view_output = gr.Textbox(label="Document Content", interactive=False)

            language_input.change(update_doc_ids, inputs=language_input, outputs=doc_id_input)

            view_button.click(view_document_by_id, inputs=[language_input, doc_id_input], outputs=[view_output])



    with gr.Tab("Document Index"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Document Index</h3>")

            index_output = gr.Textbox(label="Document Index", interactive=False)

            gr.Button("Show Index").click(show_index, outputs=index_output)



    with gr.Tab("Search Documents"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Search Documents</h3>")

            language_input = gr.Dropdown(choices=['ar', 'en', 'ru'], label="Select Language")

            search_input = gr.Textbox(label="Search Term", placeholder="Enter text to search")

            search_button = gr.Button("Search", variant="secondary")

            search_output = gr.Textbox(label="Search Results", interactive=False)

            search_button.click(search_documents, inputs=[language_input, search_input], outputs=[search_output])



    with gr.Tab("Classification Results"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Classification Results</h3>")

            classify_output = gr.Textbox(label="Classification Results", interactive=False)

            gr.Button("Show Classifications").click(show_classifications, outputs=classify_output)



    with gr.Tab("About"):

        gr.Markdown("""

        <div style='text-align: center;'>

            <h3>Smart Archiving System</h3>

            <p>Version: 3.0 (Document Management & More)</p>

            <p>Developed for intelligent document archiving with enhanced features including document deletion, download, browsing by category, and viewing documents by ID. It relies on a combination of deep learning (DistilBERT) and machine learning (SVM, RF) models for document classification.</p>

        </div>

        """)



    with gr.Tab("Logout"):

        with gr.Column():

            gr.Markdown("<h3 style='color: #555;'>Logout</h3>")

            def logout():

                current_user["username"] = None

                current_user["role"] = None

                return "**Logged out** successfully."

            logout_output = gr.Textbox(label="Status", value="")

            gr.Button("Logout").click(logout, outputs=logout_output)


if __name__ == "__main__":
    demo.launch()
