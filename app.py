import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import os
import speech_recognition as sr
from transformers import pipeline
import io

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the emotion detection model (for emotion analysis on recognized text)
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Initialize FAISS index globally (we'll populate it later)
index = faiss.IndexFlatL2(384)  # For the 'all-MiniLM-L6-v2' model, embedding size is 384
documents = []

# Use session_state to persist conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Initialize conversation history

# Function to process documents and create embeddings
def process_documents(uploaded_files):
    global documents, index
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Read the content of the uploaded file
        file_content = uploaded_file.read().decode("utf-8")  # Read and decode the file to text
        
        # Split the content by lines, assuming each line is a document
        document_lines = file_content.splitlines()  
        
        # Add the document lines to the global documents list
        documents.extend(document_lines)
        
    # Create embeddings for the documents
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    
    # Add to FAISS index
    index.add(embeddings)

# Function to manually input documents (text input)
def input_documents_manually():
    global documents, index
    # User input for documents
    user_input = st.text_area("Manually enter your document(s) (one per line):", height=150)
    if user_input:
        document_lines = user_input.splitlines()
        documents.extend(document_lines)
        
        # Create embeddings for the documents
        embeddings = model.encode(documents)
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        index.add(embeddings)
        st.success(f"{len(document_lines)} documents have been added manually.")
        
# Function to get response based on user query
def query_response(user_query):
    query_embedding = model.encode([user_query])
    query_embedding = np.array(query_embedding).astype('float32')

    D, I = index.search(query_embedding, k=1)  # Find most similar document
    response = documents[I[0][0]]  # Get the corresponding document
    return response

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  

# Function to recognize speech and convert it to text
def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        return query
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Sorry, there was an issue with the speech recognition service."

# Emotion detection function
def detect_emotion(text):
    emotion = emotion_model(text)[0]['label']
    return emotion

# Store conversation history (query, response) and embeddings for fast search
def store_conversation(user_query, bot_response):
    # Store the conversation pair in session_state
    st.session_state.conversation_history.append({'user': user_query, 'response': bot_response})
    
    # Store the embeddings of both the user query and bot response
    conversation_embeddings = model.encode([user_query, bot_response])
    conversation_embeddings = np.array(conversation_embeddings).astype('float32')
    
    # Add embeddings to FAISS index
    index.add(conversation_embeddings)

# Streamlit UI setup
st.title('Conversational Agent TTS & STT With Vector Database and Emotion recognition')

# Step 1: Choose how to input documents
document_input_method = st.radio("Choose how to input documents:", ('Upload Documents', 'Manually Enter Documents'))

# Step 2: Handle document input based on choice
if document_input_method == 'Upload Documents':
    uploaded_files = st.file_uploader("Upload your document(s) (Text files)", type="txt", accept_multiple_files=True)

    if uploaded_files:
        st.write("Processing your documents...")
        process_documents(uploaded_files)
        st.write(f"{len(documents)} documents have been uploaded and processed successfully.")
        
        # Display the uploaded documents in the sidebar
        st.sidebar.title("Uploaded Documents")
        for idx, uploaded_file in enumerate(uploaded_files):
            st.sidebar.write(f"{idx + 1}. {uploaded_file.name}")
else:
    input_documents_manually()  # Allow manual text input

# Step 3: Choose Input Method (Text or Voice)
input_method = st.radio("Choose Input Method:", ('Text Input', 'Voice Input'))

# Handle text input queries
if input_method == 'Text Input' and documents:
    user_query = st.text_input("Ask your question:")
    if user_query:
        response = query_response(user_query)
        emotion = detect_emotion(response)
        st.write(f"Response: {response}")
        st.write(f"Detected Emotion: {emotion}")
        
        # Store the conversation history
        store_conversation(user_query, response)
        
        text_to_speech(response)  # Convert response to speech

# Handle voice input queries
elif input_method == 'Voice Input' and documents:
    if st.button("Click to Speak"):
        user_query = speech_to_text()
        if user_query:
            st.write(f"You said: {user_query}")
            response = query_response(user_query)
            emotion = detect_emotion(response)
            st.write(f"Response: {response}")
            st.write(f"Detected Emotion: {emotion}")
            
            # Store the conversation history
            store_conversation(user_query, response)
            
            text_to_speech(response)  # Convert response to speech

# Display the conversation context
st.subheader('Conversation History')

if st.session_state.conversation_history:
    for idx, conversation in enumerate(st.session_state.conversation_history):
        st.write(f"**{idx+1}. User**: {conversation['user']}")
        st.write(f"**Bot**: {conversation['response']}")
else:
    st.write("No conversations yet.")