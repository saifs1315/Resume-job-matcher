import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json # Import json for parsing LLM structured output

# --- Configuration ---
# Model name for sentence transformers for embeddings.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLM model name for reasoning.
LLM_MODEL_NAME = "gemini-2.0-flash"
# Chunk size for text splitting: How many characters per chunk.
CHUNK_SIZE = 1000
# Chunk overlap for text splitting: How many characters overlap between chunks.
CHUNK_OVERLAP = 200
# Top-k similar documents to retrieve from the vector store for both scoring and LLM analysis.
TOP_K_RETRIEVAL = 5

# Set up API key for Google Generative AI (if not automatically provided by the Canvas environment)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    if "GOOGLE_API_KEY" not in os.environ:
        st.warning("GOOGLE_API_KEY environment variable not found. "
                   "If running locally, please set it in your environment "
                   "or via Streamlit's secrets management for deployment. "
                   "In Canvas, it is usually auto-injected.")

# --- Helper Functions ---

@st.cache_resource
def get_embeddings_model():
    """
    Initializes and caches the HuggingFaceEmbeddings model.
    Using st.cache_resource to avoid re-loading the model every time the app reruns.
    """
    st.info("Loading embedding model... This might take a moment.", icon="⏳")
    try:
        # Use the HuggingFaceEmbeddings from the new package
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.success("Embedding model loaded successfully!")
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()


@st.cache_resource
def get_llm_model():
    """
    Initializes and caches the Google Generative AI LLM.
    """
    st.info(f"Initializing LLM: {LLM_MODEL_NAME}...", icon="🤖")
    try:
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("Cannot initialize LLM: GOOGLE_API_KEY is not set.")
            return None

        # Configure the LLM to return JSON output
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=0.2, # Lower temperature for more focused output
            # Configure response schema for structured output
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "overallScore": {"type": "integer", "description": "Overall match score between 0 and 100."},
                        "overallCompatibility": {"type": "string", "description": "A short summary (e.g., 'Excellent Fit', 'Good Alignment', 'Some Gaps', 'Limited Match')."},
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific points where the resume directly addresses job requirements."
                        },
                        "areasForDevelopment": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key requirements from the job description weakly represented or missing."
                        },
                        "tailoringSuggestions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Concrete advice on how to improve the resume for this specific job."
                        }
                    },
                    "required": ["overallScore", "overallCompatibility", "strengths", "areasForDevelopment", "tailoringSuggestions"]
                }
            }
        )
        st.success("LLM initialized with JSON output configuration!")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}. Make sure GOOGLE_API_KEY is correctly set and model supports JSON output.")
        st.stop()


def extract_text_from_pdf(pdf_file_bytes: BytesIO) -> str:
    """
    Extracts text from an uploaded PDF file.
    Args:
        pdf_file_bytes: BytesIO object of the uploaded PDF file.
    Returns:
        A string containing all extracted text from the PDF.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file_bytes)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def process_resume(resume_text: str, embeddings) -> FAISS:
    """
    Splits the resume text into chunks, generates embeddings,
    and creates a FAISS in-memory vector store.
    Args:
        resume_text: The full text content of the resume.
        embeddings: The initialized HuggingFaceEmbeddings model.
    Returns:
        A FAISS vector store containing the embedded resume chunks.
    """
    if not resume_text:
        st.warning("Resume text is empty. Cannot process.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    docs = [Document(page_content=resume_text, metadata={"source": "resume"})]
    resume_chunks = text_splitter.split_documents(docs)

    if not resume_chunks:
        st.warning("No chunks generated from resume. Check chunk size/overlap.")
        return None

    st.info(f"Splitting resume into {len(resume_chunks)} chunks...", icon="✂️")

    try:
        vector_store = FAISS.from_documents(resume_chunks, embeddings)
        st.success("Resume processed and indexed!")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_llm_match_analysis(job_description: str, relevant_resume_chunks: list[Document], llm) -> dict:
    """
    Uses the LLM to generate a match analysis in a structured JSON format.
    Args:
        job_description: The text content of the job description.
        relevant_resume_chunks: A list of Langchain Document objects, containing the most relevant resume sections.
        llm: The initialized LLM model configured for JSON output.
    Returns:
        A dictionary containing the LLM's structured analysis, or None on error.
    """
    if not llm:
        st.error("LLM not initialized or API key is missing. Cannot provide analysis.")
        return None

    resume_context = "\n\n".join([doc.page_content for doc in relevant_resume_chunks])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a highly skilled AI assistant specializing in career advisory, particularly resume and job description matching. "
             "Your task is to analyze a candidate's resume content against a job description. "
             "Provide a comprehensive analysis including an overall match score (0-100) and actionable advice. "
             "Your output MUST be valid JSON, conforming to the specified schema."
             ),
            ("user",
             f"Here is a job description:\n\n---\n{job_description}\n---\n\n"
             f"Here are the most relevant sections from a candidate's resume based on a similarity search:\n\n---\n{resume_context}\n---\n\n"
             "Please provide a comprehensive match analysis in JSON format. The JSON should have the following keys:\n"
             "  - `overallScore`: An integer between 0 and 100 representing the overall match score.\n"
             "  - `overallCompatibility`: A short summary string (e.g., 'Excellent Fit', 'Good Alignment', 'Some Gaps', 'Limited Match').\n"
             "  - `strengths`: A list of strings, detailing specific points where the resume directly addresses job requirements."
             "  - `areasForDevelopment`: A list of strings, identifying key requirements from the job description that are weakly represented or missing."
             "  - `tailoringSuggestions`: A list of strings, offering concrete advice on how the candidate could improve their resume content for this specific job."
             "Ensure the score accurately reflects the overall match, and all lists are populated with relevant points. Be professional and encouraging."
             "The *entire* response must be a single, valid JSON object, and nothing else." # Added strong instruction
            ),
        ]
    )

    try:
        st.info("Generating detailed AI analysis... This may take a moment.", icon="✍️")
        chain = prompt_template | llm
        response = chain.invoke({}) # This returns a Langchain message object

        parsed_analysis = None
        # Check if response.content is already a dict (structured output from LLM directly)
        if isinstance(response.content, dict):
            parsed_analysis = response.content
        elif isinstance(response.content, str): # The response.content is directly the string
            # Try to remove common prefixes like "json\n" or "```json\n"
            json_string = response.content.strip()
            # Remove "json\n" if it's at the start
            if json_string.startswith("json\n"):
                json_string = json_string[len("json\n"):].strip()
            # Remove markdown code block fences if they exist
            if json_string.startswith("```json"):
                json_string = json_string[len("```json"):].strip()
                if json_string.endswith("```"):
                    json_string = json_string[:-len("```")].strip()

            try:
                parsed_analysis = json.loads(json_string)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing LLM response as JSON: {e}. Raw LLM output: {json_string}")
                print(f"DEBUG: Raw LLM output leading to JSON error: {json_string}") # For console debugging
                return None
        else:
            # If neither content (dict) nor text (string) is as expected
            st.error(f"Unexpected LLM response format. Raw response type: {type(response.content)}. Raw response: {response}")
            print(f"DEBUG: Unexpected LLM response format: {response}") # For console debugging
            return None

        # Validate basic structure after parsing
        if parsed_analysis and all(k in parsed_analysis for k in ["overallScore", "overallCompatibility", "strengths", "areasForDevelopment", "tailoringSuggestions"]):
            return parsed_analysis
        else:
            st.error("LLM returned valid JSON but missing expected keys. Retrying might help.")
            print(f"DEBUG: Parsed JSON missing keys: {parsed_analysis}") # For console debugging
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred during LLM analysis: {e}. Ensure LLM is configured for JSON output.")
        return None


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Resume-Job Description Matcher Bot", layout="wide")

    st.title("📄 Job Matcher Bot with AI Insights 🤖")
    st.markdown(
        """
        Upload your resume (PDF) and paste a job description. This bot will give you an AI-generated match score,
        a detailed analysis, and highlight the most relevant sections of your resume!
        """
    )

    # Initialize embedding and LLM models (cached for performance)
    embeddings = get_embeddings_model()
    llm = get_llm_model() # This now initializes the LLM with JSON schema

    if embeddings is None or llm is None:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Your Resume (PDF)")
        uploaded_resume = st.file_uploader(
            "Choose a PDF file", type="pdf", help="Please upload your resume in PDF format."
        )
        resume_text = ""
        if uploaded_resume:
            with st.spinner("Extracting text from resume..."):
                resume_text = extract_text_from_pdf(uploaded_resume)
                if resume_text:
                    st.success("Resume text extracted!")
                else:
                    st.error("Failed to extract text from resume.")
    
    with col2:
        st.subheader("2. Paste Job Description")
        job_description = st.text_area(
            "Paste the full job description here",
            height=300,
            help="Copy and paste the entire job description from a job portal."
        )

    st.markdown("---")

    if st.button("Analyze Match", use_container_width=True, type="primary"):
        if not uploaded_resume:
            st.warning("Please upload your resume first.")
        elif not job_description:
            st.warning("Please paste the job description.")
        elif not resume_text:
            st.error("Could not process resume. Please ensure it's a readable PDF.")
        else:
            with st.spinner("Processing resume and retrieving relevant sections..."):
                vector_store = process_resume(resume_text, embeddings)
                if vector_store:
                    relevant_chunks_with_scores = vector_store.similarity_search_with_score(
                        job_description, k=TOP_K_RETRIEVAL
                    )
                    
                    if relevant_chunks_with_scores:
                        # Prepare chunks for LLM (just the Document objects)
                        relevant_docs_for_llm = [doc for doc, _ in relevant_chunks_with_scores]

                        # Get LLM generated analysis
                        llm_analysis_output = get_llm_match_analysis(
                            job_description, relevant_docs_for_llm, llm
                        )
                        
                        if llm_analysis_output:
                            # Extract score and analysis from LLM's structured output
                            numerical_match_score = llm_analysis_output.get("overallScore", 0)
                            overall_compatibility = llm_analysis_output.get("overallCompatibility", "N/A")
                            strengths = llm_analysis_output.get("strengths", [])
                            areas_for_development = llm_analysis_output.get("areasForDevelopment", [])
                            tailoring_suggestions = llm_analysis_output.get("tailoringSuggestions", [])

                            st.subheader(f"✨ Overall Match Score: {numerical_match_score:.0f}%")
                            st.progress(numerical_match_score / 100)
                            
                            st.markdown(f"**Overall Compatibility:** {overall_compatibility}")

                            st.markdown("---")
                            st.subheader("📊 Detailed AI Analysis:")
                            st.markdown("**Strengths:**")
                            for s in strengths:
                                st.markdown(f"- {s}")
                            
                            st.markdown("**Areas for Development:**")
                            for a in areas_for_development:
                                st.markdown(f"- {a}")

                            st.markdown("**Tailoring Suggestions:**")
                            for t in tailoring_suggestions:
                                st.markdown(f"- {t}")
                            
                            st.markdown("---")
                            st.subheader("🔍 Most Relevant Resume Sections (with their individual relevance):")
                            for i, (doc, score) in enumerate(relevant_chunks_with_scores):
                                # Clamp score between -1 and 1 if floating point inaccuracies occur
                                score = max(-1.0, min(1.0, score))
                                # Convert cosine similarity to a 0-100% scale for display
                                percentage_score = ((score + 1) / 2) * 100
                                st.markdown(f"**Section {i+1} - Relevance: {percentage_score:.2f}%**")
                                st.info(doc.page_content)
                                st.markdown("---")
                        else:
                            st.error("Failed to get a structured analysis from the AI. Please try again.")

                    else:
                        st.warning("No highly relevant sections found in the resume for the job description. "
                                   "The AI analysis might be limited or generic.")
                        # Still attempt LLM analysis, but with no relevant chunks, it will be very general
                        llm_analysis_output = get_llm_match_analysis(job_description, [], llm)
                        if llm_analysis_output:
                            numerical_match_score = llm_analysis_output.get("overallScore", 0)
                            overall_compatibility = llm_analysis_output.get("overallCompatibility", "N/A")
                            strengths = llm_analysis_output.get("strengths", [])
                            areas_for_development = llm_analysis_output.get("areasForDevelopment", [])
                            tailoring_suggestions = llm_analysis_output.get("tailoringSuggestions", [])

                            st.subheader(f"✨ Overall Match Score: {numerical_match_score:.0f}% (Limited Context)")
                            st.progress(numerical_match_score / 100)
                            st.markdown(f"**Overall Compatibility:** {overall_compatibility}")
                            st.markdown("**Strengths:**")
                            for s in strengths: st.markdown(f"- {s}")
                            st.markdown("**Areas for Development:**")
                            for a in areas_for_development: st.markdown(f"- {a}")
                            st.markdown("**Tailoring Suggestions:**")
                            for t in tailoring_suggestions: st.markdown(f"- {t}")
                        else:
                            st.error("Failed to get any analysis from the AI.")


    st.markdown("---")
    st.caption("Powered by Langchain, Streamlit, Hugging Face Embeddings, and Google Gemini.")

if __name__ == "__main__":
    main()