
# Conversational Agent with TTS, STT, Emotion Detection, and Vector Database

## Overview
This project implements a conversational agent using a combination of advanced technologies for speech, text, and emotion-based interactions. The agent integrates FAISS for vector similarity search, Streamlit for an interactive user interface, and pre-trained models for natural language processing and emotion detection.

## Features
1. **Text-to-Speech (TTS):** Converts bot responses into audio output for users.
2. **Speech-to-Text (STT):** Transforms user voice input into text queries.
3. **Emotion Detection:** Identifies emotions in the responses for context-aware conversations.
4. **Vector Database (FAISS):** Handles embeddings for documents and queries, enabling similarity-based retrieval.
5. **Streamlit Interface:** Offers an intuitive and interactive user interface.
6. **Conversation History:** Tracks previous interactions for better context management.

## Technology Stack
- **Programming Language:** Python
- **Libraries and APIs:**
  - **Streamlit:** For building the web-based interface.
  - **FAISS:** For efficient vector similarity searches.
  - **SentenceTransformer:** To generate embeddings for text documents and queries.
  - **Transformers:** Pre-trained models for emotion classification.
  - **gTTS:** For converting text to speech.
  - **SpeechRecognition:** For recognizing and transcribing user speech.
- **Pre-trained Models:**
  - `all-MiniLM-L6-v2` for text embeddings.
  - `j-hartmann/emotion-english-distilroberta-base` for emotion detection.

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <https://github.com/Vatsa10/Conversational-Agent>
   cd <\Conversational-Agent>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. **Document Input:**
   - Upload text files or manually input documents to populate the FAISS index.
   - Each line of the input is treated as a separate document.

2. **Query Input:**
   - Select Text or Voice input.
   - Submit queries to retrieve the most relevant document based on similarity.

3. **Response and Emotion Detection:**
   - Displays the retrieved document as a response.
   - Shows the emotion detected in the response.

4. **Text-to-Speech:**
   - Converts the bot's response into audio for enhanced interactivity.

5. **Conversation History:**
   - Maintains a record of previous interactions for context.

## Future Enhancements
- Add multi-language support for TTS and STT.
- Integrate advanced emotion detection with visual cues (e.g., facial expressions).
- Enhance conversational logic with memory and user personalization.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [SentenceTransformers](https://www.sbert.net/) for semantic text embeddings.
- [Hugging Face](https://huggingface.co/) for pre-trained NLP models.
- [Streamlit](https://streamlit.io/) for creating an easy-to-use UI framework.
