import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer("all-MiniLM-L6-v2")
def compute_similarity(sentences1, sentences2):
    # Encode the lists to get embeddings
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    # Compute cosine similarity
    cosine_scores = util.cos_sim(embeddings1,embeddings2)
    return cosine_scores

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
task = st.sidebar.selectbox("Choose a task:", ["Translate", "Summarize", "Classify","Find Similarity"])

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

elif task == "Find Similarity":
    st.subheader("Find Similarity Between Two Lists")
    list1_input = st.text_area("Enter the first list (comma-separated):")
    list2_input = st.text_area("Enter the second list (comma-separated):")
    
    if st.button("Find Similarity"):
        if list1_input and list2_input:
            list1 = [item.strip() for item in list1_input.split(",")]
            list2 = [item.strip() for item in list2_input.split(",")]
            similarities = compute_similarity(list1, list2)
            for i in range(len(list1)):
                st.write("Similarity between {} \t\t and  {} \t\t Score: {:.4f}".format(list1[i],
                                                 list2[i],
                                                 similarities[i][i]))
        else:
            st.warning("Please enter both lists to compare.")
# The dog plays in the garden, A woman watches TV, The new movie is so great
# The cat sits outside, A man is playing guitar, The movies are awesome