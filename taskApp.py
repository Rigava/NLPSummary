import streamlit as st
import torch
from transformers import pipeline



# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

# Initialize other pipelines
summarizer = pipeline(task="summarization",
                      model="facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16
                      )
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# translator = pipeline(task="translation",
#                       model="facebook/nllb-200-distilled-600M"
#                       ) 

# Set up the Streamlit app
st.title("NLP Task Switcher")
st.sidebar.title("Select Task")

# Create a sidebar to select the task
task = st.sidebar.selectbox("Choose a task:", ["Summarize", "Classify","Translate"])

if  task == "Summarize":
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
elif task == "Translate":
    st.subheader("Translation")
    text_to_translate = st.text_area("Enter text to translate:")
    # Initialize translation pipelines for each language
    translation_pipelines = {
    "French": translator(text_to_translate,src_lang="eng_Latn",tgt_lang="fra_Latn"),
    "Hindi": translator(text_to_translate,src_lang="eng_Latn",tgt_lang="hin_Deva"),

}
    target_language = st.selectbox("Select target language:", list(translation_pipelines.keys()))
    
    if st.button("Translate"):
        if text_to_translate:
            translation = translation_pipelines[target_language][0]['translation_text']
            st.write("Translation to {}: {}".format(target_language, translation))
        else:
            st.warning("Please enter text to translate.")

# The cat sits outside, A man is playing guitar, The movies are awesome
