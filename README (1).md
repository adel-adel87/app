# Model artifacts

The trained models are not included in this repository.

Place the artifacts exported by your training notebook here:
- `svm_model.joblib`
- `rf_model.joblib`
- `vectorizer.joblib`
- `le.joblib`
- `distilbert_model/` (Hugging Face model directory)
- `distilberttokenizer_model/` (tokenizer directory)

Use trusted model files only. `joblib.load()` can execute code from malicious pickle files; do not load artifacts from untrusted sources. The app starts without these files, but document classification remains unavailable until they are installed.
