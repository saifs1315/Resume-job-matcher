📄 Resume & Job Description Matcher Bot with AI Insights 🤖
This project is an interactive web application designed to help job seekers assess how well their resume aligns with a specific job description. Leveraging advanced Natural Language Processing (NLP) and Large Language Models (LLMs), it provides both a numerical match score and a detailed AI-generated analysis, offering actionable insights for tailoring your application.

This bot is built entirely using free and open-source technologies, making it an excellent portfolio piece.

✨ Features
PDF Resume Upload: Easily upload your resume in PDF format.

Job Description Input: Paste the job description directly into the application.

AI-Generated Match Score: Get an overall compatibility score (0-100%) for your resume against the job description, powered by an LLM's reasoning.

Detailed AI Analysis: Receive a comprehensive breakdown from the AI, including:

Overall compatibility assessment (e.g., "Good Alignment", "Some Gaps").

Key strengths identified in your resume.

Areas for development or missing requirements.

Actionable tailoring suggestions to improve your resume for that specific job.

Relevant Sections Highlighting: See the most relevant parts of your resume that the AI considered for its analysis, along with their individual relevance scores.

🛠️ Technologies Used
Python: The core programming language.

Streamlit: For building the interactive web user interface.

Langchain: Framework for orchestrating LLM interactions, document loading, and text processing.

Hugging Face sentence-transformers: For generating high-quality semantic embeddings from text (runs locally, no API key needed).

FAISS: For efficient in-memory similarity search over resume content.

pypdf: For extracting text from PDF resume files.

Google Gemini API (gemini-2.0-flash): The LLM used for generative analysis and score calculation (free tier available).

🚀 Getting Started
Follow these instructions to set up and run the application locally or deploy it online.

Prerequisites
Python 3.8+ installed on your system.

A Google account to obtain a Gemini API key.

Local Setup & Run
Clone the Repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

(Replace YOUR_USERNAME and YOUR_REPOSITORY_NAME with your actual GitHub details.)

Create a Virtual Environment:

python -m venv venv

Activate the Virtual Environment:

macOS / Linux:

source venv/bin/activate

Windows (Command Prompt):

.\venv\Scripts\activate

Windows (PowerShell):

.\venv\Scripts\Activate.ps1

Install Dependencies:
Make sure you have a requirements.txt file in your repository with the following content:

streamlit
langchain-community>=0.0.30
langchain-core>=0.1.30
langchain-google-genai>=0.0.10
langchain-huggingface
pypdf
faiss-cpu
sentence-transformers
google-generativeai

Then install them:

pip install -r requirements.txt

Get Your Google Gemini API Key:

Go to Google AI Studio and log in with your Google account.

Follow the prompts to "Get API key in Google AI Studio" and create a new API key.

Copy the generated key.

Set the API Key as an Environment Variable:
Before running the app, set your API key in your terminal session. Replace "YOUR_ACTUAL_GOOGLE_API_KEY_HERE" with your copied key.

macOS / Linux:

export GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Windows (Command Prompt):

set GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Windows (PowerShell):

$env:GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Run the Streamlit App:

streamlit run app.py

The application will open in your default web browser (usually at http://localhost:8501).

Online Deployment (Streamlit Community Cloud)
You can deploy this application for free using Streamlit Community Cloud.

Ensure GitHub Repository is Ready:

Your app.py and requirements.txt files must be committed and pushed to a public GitHub repository.

Securely Add Your API Key:

Do NOT include your GOOGLE_API_KEY directly in your app.py file or secrets.toml within your public GitHub repository.

Go to share.streamlit.io and sign in with your GitHub account.

When deploying your app, or in its settings, navigate to the "Advanced settings" or "Secrets" section.

Click "Add a new secret".

Set the Key as GOOGLE_API_KEY and the Value as your actual Gemini API key. Save this secret.

Deploy the App:

On the Streamlit Community Cloud dashboard, click "New app".

Select your GitHub repository, the correct branch (e.g., main), and the path to your app.py file.

Confirm your GOOGLE_API_KEY is set in the secrets.

Click "Deploy!".

Streamlit Community Cloud will automatically build and host your application, providing you with a public URL. Any future pushes to your selected GitHub branch will automatically update your deployed app.

👩‍💻 Usage
Upload Your Resume: Click the "Browse files" button to upload your resume in PDF format.

Paste Job Description: Copy and paste the full job description text into the provided text area.

Analyze Match: Click the "Analyze Match" button.

View Results: The app will display an overall match score, a detailed AI-generated analysis (strengths, areas for improvement, and tailoring suggestions), and the most relevant sections of your resume that the AI considered.

🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the bot's functionality, accuracy, or user experience are welcome!

📄 License
This project is open-source and available under the MIT License (or choose your preferred open-source license).