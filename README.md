# Gemini Quizify

Gemini Quizify generates quizzes from documents with various filetype support. It is designed for educators and students with an accessible and effective means to reinforce their understanding of any topic by offering instant feedback and detailed explanations, ultimately enhancing the learning experience.

Tech stack: GCP (Vertex AI), Streamlit, Langchain, ChromaDB

## Features
- **Dynamic Quiz Generation**: Create customizable quizzes from PDFs tailored to your needs.
- **Instant Feedback**: Receive explanations for each quiz question and feedback on your answers to assist in understanding.
- **User-Friendly Interface**: Easily navigate through the quiz with a Streamlit user interface.

## Prerequisites
- Python 3.7 or higher
- Google Cloud account with Vertex AI enabled

## Setup
1. Clone the Repository
```bash
git clone https://github.com/lancegosu/quizify.git
cd quizify
```

2. Set Up Google Cloud and Vertex AI
- Create a Google Cloud account and project.
- Enable Vertex AI APIs.
- Create a service account and download the key JSON file.
- Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the key JSON file.
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"
```

3. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
1. Start the Streamlit Application
```bash
streamlit run quizify.py
```

2. Upload a Document
- Use the file uploader to upload a PDF document.

3. Select Quiz Parameters
- Choose the topic and number of questions for the quiz.

4. Generate Quiz
- Click the button to generate the quiz.
- View the generated quiz questions and provide answers.
- Receive instant feedback and detailed explanations for each question.

5. Navigate the Quiz
- Use the interface to navigate through the quiz questions.

By following this guide, you should be able to set up and use Gemini Quizify to generate dynamic quizzes based on user-provided documents. Enjoy enhancing your learning experience with AI-powered quizzes!

### Acknowledgements
A special thank you to Radical AI for providing the foundation of this project. Their support and resources have been invaluable in developing Gemini Quizify, ensuring that we can offer a robust and effective learning tool for educators and students worldwide.
