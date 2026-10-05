import joblib
from utils import extract_text_from_pdf, preprocess_text

# Load trained models
vec = joblib.load("models/tfidf_vectorizer.joblib")
knn = joblib.load("models/knn.joblib")

# Your resume path
pdf_path = r"C:\Users\GOKUL\OneDrive\Desktop\gkresume.pdf"

# Extract + preprocess text
text = extract_text_from_pdf(pdf_path)
clean = preprocess_text(text)

# Predict
vec_text = vec.transform([clean])
print(vec_text)
pred = knn.predict(vec_text)[0]

print("\nPredicted Category:", pred)
