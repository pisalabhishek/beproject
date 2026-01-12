import numpy as np
import re
import spacy
import torch
from pdfminer.high_level import extract_text
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile


st.set_page_config(
    page_title="AI-Powered Resume Ranking",
    layout="wide"
)


# Load NLP models
@st.cache_resource
def load_models():
    nlp = spacy.load('en_core_web_sm')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return nlp, tokenizer, model


nlp, tokenizer, model = load_models()


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        text = extract_text(tmp_file_path)
        if not text.strip():
            raise ValueError("Empty or unreadable PDF.")
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# Clean Resume
def clean_resume(text):
    text = re.sub(r'\W+', ' ', text)
    doc = nlp(text.lower())
    clean_text = ' '.join([token.lemma_ for token in doc if token.text not in nlp.Defaults.stop_words])
    return clean_text


# Get BERT Embeddings (Batch Processing)
def get_bert_embedding(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


# Rank Resumes using multiple models
def rank_resumes(job_desc_embedding, resume_embeddings, tfidf_matrix, tfidf_vectorizer, job_desc_text):
    similarity_scores = cosine_similarity(job_desc_embedding, resume_embeddings).flatten()
    tfidf_scores = tfidf_matrix @ tfidf_vectorizer.transform([job_desc_text]).T
    tfidf_scores = tfidf_scores.toarray().flatten()
    final_scores = (0.7 * similarity_scores) + (0.3 * tfidf_scores / tfidf_scores.max())

    # Prepare training data
    X = final_scores.reshape(-1, 1)
    y = [1 if score >= 0.5 else 0 for score in final_scores]

    model_accuracies = {}

    if len(set(y)) > 1:  # Ensure we have both classes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        }

        best_model = None
        best_accuracy = 0

        for name, ml_model in models.items():
            ml_model.fit(X_train, y_train)
            acc = accuracy_score(y_test, ml_model.predict(X_test))
            model_accuracies[name] = acc
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = ml_model

    ranked_indices = np.argsort(final_scores)[::-1]
    return ranked_indices, final_scores[ranked_indices], model_accuracies


# Save results to CSV
def save_results_to_csv(ranked_results):
    return ranked_results.to_csv(index=False).encode('utf-8')


# Graphical Visualization
def plot_results(ranked_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ranked_results['Resume'], ranked_results['Score'].astype(float), color='skyblue')
    ax.set_xlabel("Score")
    ax.set_ylabel("Resumes")
    ax.set_title("Resume Ranking Scores")
    ax.invert_yaxis()
    return fig


# Process Resumes
def process_resumes(resume_files, job_desc_text):
    resume_embeddings = []
    cleaned_resumes = []
    resume_names = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, resume_file in enumerate(resume_files):
        try:
            status_text.text(f"Processing {resume_file.name}...")
            pdf_text = extract_text_from_pdf(resume_file)
            clean_text = clean_resume(pdf_text)
            cleaned_resumes.append(clean_text)
            resume_names.append(resume_file.name)
            progress_bar.progress((idx + 1) / len(resume_files))
        except Exception as e:
            st.error(f"Error processing {resume_file.name}: {str(e)}")
            return None, None, None

    status_text.empty()
    progress_bar.empty()

    if not cleaned_resumes:
        st.error("No valid resumes found.")
        return None, None, None

    status_text.text("Generating embeddings...")
    resume_embeddings = get_bert_embedding(cleaned_resumes)
    job_desc_embedding = get_bert_embedding([job_desc_text])

    status_text.text("Calculating TF-IDF scores...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_resumes)

    status_text.text("Ranking resumes...")
    ranked_indices, scores, model_accuracies = rank_resumes(job_desc_embedding, resume_embeddings, tfidf_matrix,
                                                            tfidf_vectorizer, job_desc_text)

    ranked_results = pd.DataFrame({
        "Rank": range(1, len(ranked_indices) + 1),
        "Resume": [resume_names[i] for i in ranked_indices],
        "Score": [f"{scores[i]:.4f}" for i in ranked_indices]
    })

    status_text.empty()

    return ranked_results, model_accuracies, plot_results(ranked_results)


# Streamlit UI
def main():
    #st.set_page_config(page_title="AI-Powered Resume Ranking", layout="wide")

    st.title("🎯 AI-Powered Resume Ranking with ML Models")
    st.markdown("Upload resumes and enter a job description to rank the resumes based on relevance using ML models.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Description")
        job_desc_text = st.text_area(
            "Enter the job description",
            height=300,
            placeholder="Paste the job description here..."
        )

    with col2:
        st.subheader("Upload Resumes")
        resume_files = st.file_uploader(
            "Upload PDF resumes",
            type=['pdf'],
            accept_multiple_files=True
        )

    if st.button("🚀 Rank Resumes", type="primary"):
        if not job_desc_text:
            st.warning("Please enter a job description.")
        elif not resume_files:
            st.warning("Please upload at least one resume.")
        else:
            with st.spinner("Processing resumes..."):
                ranked_results, model_accuracies, fig = process_resumes(resume_files, job_desc_text)

            if ranked_results is not None:
                st.success("✅ Ranking completed!")

                # Display results
                st.subheader("📊 Ranking Results")
                st.dataframe(ranked_results, use_container_width=True)

                # Display model accuracies
                if model_accuracies:
                    st.subheader("🤖 ML Model Accuracies")
                    acc_df = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Accuracy"])
                    acc_df["Accuracy"] = acc_df["Accuracy"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(acc_df, use_container_width=True)

                # Display chart
                st.subheader("📈 Visualization")
                st.pyplot(fig)

                # Download button
                csv_data = save_results_to_csv(ranked_results)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name="ranked_resumes.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":

    main()

