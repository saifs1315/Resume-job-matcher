📄 Resume & Job Description Matcher Bot with AI Insights 🤖
An interactive, AI-powered web app to help job seekers evaluate how well their resume aligns with a specific job description. This tool combines Natural Language Processing (NLP) and Large Language Models (LLMs) to deliver a match score and AI-generated insights to help tailor applications for better results.

Built entirely with free and open-source technologies, it’s perfect as a portfolio project or practical career tool.
✨ Features
•	📄 PDF Resume Upload: Easily upload your resume in PDF format.
•	📝 Job Description Input: Paste any job description directly into the application.
•	📊 AI-Generated Match Score: Get a 0–100% compatibility score using LLM-based reasoning.
•	🧠 Detailed AI Analysis: Receive comprehensive breakdowns, including:
- Compatibility assessment (e.g., "Good Alignment", "Some Gaps")
- Key strengths in your resume
- Missing qualifications or skills
- Actionable suggestions to tailor your resume
•	📌 Relevance Highlights: See the most relevant resume sections that influenced the AI’s analysis, each with individual relevance scores.
🛠️ Technologies Used
•	Python – Core language
•	Streamlit – UI for interactive web app
•	LangChain – LLM orchestration and text processing
•	Hugging Face sentence-transformers – Local semantic embeddings
•	FAISS – Fast in-memory similarity search
•	pypdf – PDF text extraction
•	Google Gemini API (gemini-2.0-flash) – Free-tier LLM for analysis and scoring
🚀 Getting Started
✅ Prerequisites
• Python 3.8+
• Google account for Gemini API key
🔧 Local Setup
1.	Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
2.	Create and activate a virtual environment:

python -m venv venv
# Activate
# macOS / Linux
source venv/bin/activate
# Windows CMD
.\venv\Scripts\activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1
3.	Install dependencies:

pip install -r requirements.txt
4.	Get your Gemini API key from Google AI Studio and copy it.
5.	Set the API key as an environment variable:

# macOS / Linux
export GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

# Windows CMD
set GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

# Windows PowerShell
$env:GOOGLE_API_KEY="YOUR_ACTUAL_GOOGLE_API_KEY_HERE"
6.	Run the app:

streamlit run app.py

Visit http://localhost:8501 in your browser.
🌐 Online Deployment with Streamlit Community Cloud
7.	Prepare your GitHub repo with app.py and requirements.txt committed.
8.	Secure your API key:

- Go to Streamlit Cloud
- Log in with GitHub
- Go to Settings > Secrets
- Add a new secret with key: GOOGLE_API_KEY and your Gemini API key as value.
9.	Deploy:

- Click 'New app'
- Select repo, branch, and path
- Click 'Deploy!'
👩‍💻 How to Use
•	Upload Resume – Upload a PDF of your resume.
•	Paste Job Description – Paste the text of a job posting.
•	Click “Analyze Match” – Let the AI evaluate the alignment.
•	View Results – See your match score, strengths, gaps, and suggestions.
🤝 Contributing
Contributions are welcome! Feel free to:
- Open issues
- Fork and improve the repo
- Submit pull requests

Enhancements to accuracy, functionality, and UX are greatly appreciated.
📄 License
This project is licensed under the MIT License. You’re free to use, modify, and distribute it as you like.
