import streamlit as st
from transformers import pipeline

# Initialize translation pipelines for each language
translation_pipelines = {
    "French": pipeline("translation_en_to_fr"),
    "German": pipeline("translation_en_to_de"),

}

# Initialize other pipelines
summarizer = pipeline("summarization")
classifier = pipeline("zero-shot-classification")

# Set up the Streamlit app
st.title("NLP Task Switcher")
st.sidebar.title("Select Task")

# Create a sidebar to select the task
task = st.sidebar.selectbox("Choose a task:", ["Translate", "Summarize", "Classify"])

if task == "Translate":
    st.subheader("Translation")
    text_to_translate = st.text_area("Enter text to translate:")
    target_language = st.selectbox("Select target language:", list(translation_pipelines.keys()))
    
    if st.button("Translate"):
        if text_to_translate:
            translation = translation_pipelines[target_language](text_to_translate, max_length=40)[0]['translation_text']
            st.write("Translation to {}: {}".format(target_language, translation))
        else:
            st.warning("Please enter text to translate.")

elif task == "Summarize":
    st.subheader("Summarization")
    text_to_summarize = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if text_to_summarize:
            summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.write("Summary:", summary)
        else:
            st.warning("Please enter text to summarize.")

elif task == "Classify":
    st.subheader("Text Classification")
    text_to_classify = st.text_area("Enter text to classify:")
    labels = st.text_input("Enter labels (comma-separated):", "positive, negative")
    if st.button("Classify"):
        if text_to_classify:
            labels_list = [label.strip() for label in labels.split(",")]
            classification = classifier(text_to_classify, labels_list)
            st.write("Classification:", classification)
        else:
            st.warning("Please enter text to classify.")
