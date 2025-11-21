import streamlit as st
import os
import hashlib
import tempfile
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq

# ------------------- API Keys -------------------
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "A-------YOUR API KEY--------E")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "A-------YOUR API KEY--------o")
os.environ["GOOGLE_CSE_ID"] = os.environ.get("GOOGLE_CSE_ID", "c-------YOUR API KEY--------0")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "g-------YOUR API KEY--------E")

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")
st.title("📚 Multi-Agentic RAG - College Study Assistant")

# Dropdown for model selection
model_options = {
"Gemini 2.5 Flash": "gemini-2.5-flash",
"Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
"Gemini 2.0 Flash": "gemini-2.0-flash",
"GPT-OSS-120B": "openai/gpt-oss-120b"
}
selected_model_name = st.selectbox("Select Language Model", list(model_options.keys()), index=0)

# Initialize the selected model
selected_model = None
if "Gemini" in selected_model_name:
try:
selected_model = ChatGoogleGenerativeAI(
model=model_options[selected_model_name],
temperature=0.3,
google_api_key=os.environ["GEMINI_API_KEY"]
)
except Exception as e:
st.error(f"Failed to initialize Gemini model: {e}")
elif selected_model_name == "GPT-OSS-120B":
try:
selected_model = ChatGroq(
model_name=model_options[selected_model_name],
api_key=os.environ["GROQ_API_KEY"],
temperature=0.3,
max_tokens=2048
)
except Exception as e:
st.error(f"Failed to initialize Groq model: {e}")
else:
st.error("Invalid model selected.")

# ------------------- Embedding Model with Retry Logic -------------------
def create_embedding_model_with_retry(model_name="models/embedding-001", retries=3, backoff_factor=2):
for attempt in range(retries):
try:
embedding_model = GoogleGenerativeAIEmbeddings(
model=model_name,
google_api_key=os.environ["GEMINI_API_KEY"]
)
_ = embedding_model.embed_query("connectivity probe")
return embedding_model
except Exception as e:
if attempt < retries - 1:
delay = backoff_factor ** attempt
st.warning(f"Embedding API attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} sec...")
time.sleep(delay)
else:
raise Exception(f"Failed to initialize embedding model after {retries} attempts: {str(e)}")

embedding_model = None
try:
embedding_model = create_embedding_model_with_retry()
except Exception as e:
st.error(f"Error initializing embedding model: {str(e)}. Check GEMINI_API_KEY or network.")

# ------------------- File Uploader -------------------
uploaded_files = st.file_uploader("📄 Upload your study material (PDFs)", type=["pdf"], accept_multiple_files=True)
retriever = None
VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

def get_pdf_hash(file_path):
with open(file_path, "rb") as f:
pdf_bytes = f.read()
return hashlib.md5(pdf_bytes).hexdigest()

all_dbs = []
if uploaded_files and embedding_model:
for uploaded_file in uploaded_files:
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
tmp_file.write(uploaded_file.read())
file_path = tmp_file.name
pdf_hash = get_pdf_hash(file_path)
vector_path = os.path.join(VECTOR_DIR, pdf_hash)

if os.path.exists(vector_path):
st.info(f"🔄 Reusing existing embeddings for {uploaded_file.name}...")
try:
db = FAISS.load_local(
vector_path,
embeddings=embedding_model,
allow_dangerous_deserialization=True
)
except Exception as e:
st.error(f"Error loading vector store for {uploaded_file.name}: {str(e)}. Recomputing embeddings...")
loader = PyMuPDFLoader(file_path)
documents = loader.load()
for doc in documents:
doc.metadata.update({
"source": uploaded_file.name,
"hash": pdf_hash,
"uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
db = FAISS.from_documents(docs, embedding_model)
db.save_local(vector_path)
else:
st.info(f"📚 Creating new embeddings for {uploaded_file.name}...")
loader = PyMuPDFLoader(file_path)
documents = loader.load()
for doc in documents:
doc.metadata.update({
"source": uploaded_file.name,
"hash": pdf_hash,
"uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
db = FAISS.from_documents(docs, embedding_model)
db.save_local(vector_path)
st.success(f"✅ Vector store created and saved for {uploaded_file.name}!")

all_dbs.append(db)

if all_dbs:
db = all_dbs[0]
for extra_db in all_dbs[1:]:
db.merge_from(extra_db)
retriever = db.as_retriever(search_kwargs={"k": 10})

# ------------------- Token Estimation Function -------------------
def estimate_tokens(text: str) -> int:
return max(1, len(text) // 4)

# ------------------- Tool Agents -------------------
if retriever and selected_model:
def get_combined_context(query: str, max_chunks=20) -> str:
docs = retriever.get_relevant_documents(query)
if not docs:
return ""
return "\n\n".join([f"[{doc.metadata.get('source','unknown')} p.{doc.metadata.get('page','?')}] {doc.page_content}" for doc in docs[:max_chunks]])

@tool
def summarizer(query: str, context: str = "") -> str:
"""
Summarize the topic in the user's query using ONLY the retrieved document chunks or provided context.
Args:
query: The user's instruction (may include 'in X lines').
context: Optional external context (e.g., from web search).
Returns:
A concise paragraph-level summary based strictly on the provided context.
"""
length_match = re.search(r'in\s+(\d+)\s+lines', query.lower())
desired_lines = int(length_match.group(1)) if length_match else 10
if context:
combined_content = context
else:
combined_content = get_combined_context(query)
if not combined_content:
return f"No relevant content found in the documents for '{query}'."
prompt = (
f"Summarize the following strictly from the provided document content (no outside knowledge). "
f"Write ~{desired_lines} lines, paragraph form, with simple examples where helpful:\n\n{combined_content}"
)
response = selected_model.invoke(prompt)
return getattr(response, "content", str(response))

@tool
def mcq_generator(query: str, context: str = "") -> str:
"""Generates multiple-choice questions (MCQs) based on the content with specified count."""
count_match = re.search(r'(\d+)\s*(?:mcqs|mcq|questions)', query.lower())
desired_count = int(count_match.group(1)) if count_match else 20
if context:
combined_content = context
else:
docs = retriever.get_relevant_documents(query)
if not docs:
return "No relevant content found in the document."
combined_content = "\n\n".join([doc.page_content for doc in docs])
prompt = (
f"You are a knowledgeable assistant tasked with creating multiple-choice questions (MCQs) for a learning module. "
f"Given the context below, generate {desired_count} MCQs. Number them as Question 1, Question 2, etc. "
f"If the context refers to a specific problem like 'problem 6.5', describe the problem in the question text if necessary. "
f"Each MCQ should include:\n"
f"- A single, clear question prefixed with 'Question N: ' where N is the question number\n"
f"- If the question requires a diagram (e.g., if the context describes a figure, circuit, or visual element), include a blank line after the question, then 'diagram:' on its own line, followed by a rough text-based ASCII art sketch using lines, arrows, and symbols to represent the diagram, then a blank line\n"
f"- The word 'options' on its own line\n"
f"- Four answer options labeled a), b), c), d) on separate lines, with no space after the letter\n"
f"- A blank line after the options\n"
f"- The correct answer as 'correct answer: [letter])[Option]' on its own line\n"
f"- A blank line after the correct answer\n"
f"- An explanation labeled 'explanation: ' followed by exactly two lines\n"
f"- Two blank lines after each MCQ group\n"
f"- Do not add any additional text or spaces\n"
f"- Strictly adhere to the example format, including newlines and no space after option letters\n\n"
f"Text:\n"
f"{combined_content}\n\n"
f"Example without diagram:\n"
f"Question 1: The following substance(s) is (are) ketogenic\n"
f"\n"
f"options\n"
f"a)Fatty acids\n"
f"b)Leucine\n"
f"c)Lysine\n"
f"d)All of them\n"
f"\n"
f"correct answer: d)All of them\n"
f"\n"
f"explanation:\n"
f"Fatty acids, leucine, and lysine are all ketogenic substances.\n"
f"Ketogenic substances can be converted into ketone bodies.\n"
f"\n"
f"\n"
f"Example with diagram:\n"
f"Question 2: In the circuit shown, what is the voltage across the resistor?\n"
f"\n"
f"diagram:\n"
f"+-----R----+\n"
f"| |\n"
f"V |\n"
f"| |\n"
f"+----------+\n"
f"\n"
f"options\n"
f"a)5V\n"
f"b)10V\n"
f"c)15V\n"
f"d)20V\n"
f"\n"
f"correct answer: b)10V\n"
f"\n"
f"explanation:\n"
f"The voltage source is directly across the resistor in a simple series circuit.\n"
f"Assuming V=10V and no other components.\n"
f"\n"
f"\n"
)
response = selected_model.invoke(prompt)
return response.content

@tool
def notes_maker(query: str, context: str = "") -> str:
"""
Create concise notes ONLY from the retrieved context or provided context, using side headings.
Args:
query: The user's instruction (may include 'in X lines').
context: Optional external context (e.g., from web search).
Returns:
Short paragraphs grouped by side headings like ## Key Concept, ## Formula, etc.
"""
length_match = re.search(r'in\s+(\d+)\s+lines', query.lower())
desired_lines = int(length_match.group(1)) if length_match else 20
if context:
combined_content = context
else:
combined_content = get_combined_context(query)
if not combined_content:
return f"No relevant content found in the documents for '{query}'."
prompt = (
f"You are an expert note maker specializing in creating efficient notes for competitive exams "
f"(like railways, UPSC, government job exams, semester exams, and subjective exams). "
f"Your goal is to reduce the user's time in note-making while helping them score high marks.\n\n"
f"Task: Create concise notes (~{desired_lines} lines) strictly from the provided content.\n\n"
f"Rules:\n"
f"- Use side headings like ## Key Concept, ## Formula, ## Important Fact, ## Mnemonic, ## Revision Point, ## Exam Tip\n"
f"- No bullet points; write in short paragraphs\n"
f"- Include simple examples where helpful\n"
f"- You may use simple text-based tables/diagrams for clarity\n\n"
f"Context:\n{combined_content}\n"
)
response = selected_model.invoke(prompt)
return getattr(response, "content", str(response))

@tool
def exam_prep_agent(query: str, context: str = "") -> str:
"""
Build a study plan and revision notes ONLY from the retrieved context or provided context.
Args:
query: The user's instruction.
context: Optional external context (e.g., from web search).
Returns:
A structured study plan with revision tips derived strictly from context.
"""
if context:
combined_content = context
else:
combined_content = get_combined_context(query)
if not combined_content:
return f"No relevant content found in the documents for '{query}'."
prompt = (
f"Prepare a study plan and revision notes strictly from this content:\n\n{combined_content}"
)
response = selected_model.invoke(prompt)
return getattr(response, "content", str(response))

@tool
def concept_explainer(query: str, context: str = "") -> str:
"""
Explain the requested concept in simple terms using ONLY the retrieved context or provided context.
Args:
query: The user's instruction or question (e.g., 'Explain X in 5 lines', 'What is Y').
context: Optional external context (e.g., from web search).
Returns:
A clear explanation constrained to the provided context.
"""
if context:
combined_content = context
else:
combined_content = get_combined_context(query)
if not combined_content:
return f"No relevant content found in the documents for '{query}'."
match = re.search(r'(\d+)\s*lines?', query, re.IGNORECASE)
if match:
num_lines = int(match.group(1))
line_instruction = f"Answer strictly in {num_lines} clear and concise lines."
else:
line_instruction = "Give a clear explanation in simple terms."
prompt = (
f"Explain the concept '{query}' in simple terms using ONLY the following context:\n\n"
f"{combined_content}\n\n"
f"{line_instruction}"
)
response = selected_model.invoke(prompt)
return getattr(response, "content", str(response))

@tool
def search_agent(query: str) -> dict:
"""
Use Google Custom Search to fetch up-to-date public web snippets and answer the query.
This agent is designed for students preparing for competitive exams like JEE, NEET, UPSC,
and for research topics not in the vector database. It searches the web, extracts complete information
from up to 10 websites, and determines if a subtool is needed for further processing.

Args:
query: The user's web question.
Returns:
A dictionary with 'content' (the search results) and 'subtool' (the subtool to invoke, if any).
"""
st.info("🔍 Searching the web for up-to-date information...")
search = GoogleSearchAPIWrapper(
google_api_key=os.environ["GOOGLE_API_KEY"],
google_cse_id=os.environ["GOOGLE_CSE_ID"]
)
try:
results = search.results(query, num_results=10)
full_contents = []
for res in results:
try:
response = requests.get(res['link'], timeout=5)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text(separator='\n', strip=True)[:5000]
full_contents.append(f"Title: {res.get('title', 'No title')}\nLink: {res.get('link', 'No link')}\nContent: {text}")
except Exception as fetch_e:
full_contents.append(f"Title: {res.get('title', 'No title')}\nLink: {res.get('link', 'No link')}\nSnippet: {res.get('snippet', 'No snippet')}\nError fetching full content: {str(fetch_e)}")
text = "\n\n".join(full_contents)
prompt = (
f"Query: {query}\n\n"
f"Based on this online content, extract and organize the relevant information to answer the query.\n\n"
f"Instructions:\n"
f"- Extract and display the complete relevant information from each website's content, including all key details.\n"
f"- Organize by source if multiple.\n"
f"- Include mathematical equations in LaTeX format (e.g., \\( V = IR \\)) where present.\n"
f"- If diagrams are described, represent them as ASCII art using lines, arrows, and symbols.\n"
f"- If the query asks for previous exam questions, list only the questions with equations, without theory.\n"
f"- Do not generate new content like MCQs or summaries here; just extract and organize the raw information.\n\n"
f"IMPORTANT: Analyze the FULL query '{query}' to determine the appropriate subtool for further processing. For example:\n"
f"- If the query contains 'summarize', 'summary', 'overview', 'condense', 'brief', 'key highlights', or similar, select 'summarizer'.\n"
f"- If it contains 'mcq', 'mcqs', 'questions', 'quiz', etc., select 'mcq_generator'.\n"
f"- If it contains 'notes', 'make notes', 'key points', 'revision', select 'notes_maker'.\n"
f"- If it contains 'prepare', 'exam', 'study plan', select 'exam_prep_agent'.\n"
f"- If it contains 'explain', 'what is', 'define', select 'concept_explainer'.\n"
f"- If no specific processing is needed, select 'none'.\n"
f"You MUST end your entire response with exactly this line, on its own: Final Subtool Decision: [subtool_name]\n"
f"where [subtool_name] is one of: summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer, none\n\n"
f"Content:\n{text}"
)
response = selected_model.invoke(prompt)
content = getattr(response, "content", str(response))
# Robust subtool extraction
subtool_match = re.search(r'Final Subtool Decision:\s*\[(\w+)\]', content, re.IGNORECASE)
subtool = subtool_match.group(1).lower() if subtool_match else None
# Enhanced fallback logic
if not subtool:
st.warning(f"Subtool not detected in LLM response for query: '{query}'. Using keyword fallback.")
ql = query.lower()
keywords = {
"summarizer": ["summarize", "summary", "overview", "condense", "brief", "highlights", "tl;dr"],
"mcq_generator": ["mcq", "mcqs", "question", "questions", "quiz", "practice", "test"],
"notes_maker": ["notes", "make notes", "key points", "revision", "study notes", "outline"],
"exam_prep_agent": ["prepare", "exam", "study plan", "revision plan", "strategy", "timetable"],
"concept_explainer": ["explain", "what is", "define", "how does", "breakdown", "clarify"]
}
subtool = "none"
for tool_name, tool_keywords in keywords.items():
if any(keyword in ql for keyword in tool_keywords):
subtool = tool_name
break
if subtool == "none":
st.warning(f"No subtool matched for query: '{query}'. Defaulting to 'none'.")
# Clean content
content = re.sub(r'Final Subtool Decision:.*', '', content, re.IGNORECASE | re.DOTALL).strip()
sources = "\n\nSources:\n" + "\n".join([
f"- {res.get('title', 'No title')}: {res.get('link', 'No link')}"
for res in results
])
content = content + sources if content else sources
return {"content": content, "subtool": subtool}
except Exception as e:
st.error(f"Search agent error: {str(e)}. Ensure API key and CSE ID are valid.")
return {
"content": f"Error during search: {str(e)}. Ensure API key and CSE ID are valid.",
"subtool": "none"
}

tools = [summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer, search_agent]

# Optional: bind tools if your model supports it
if hasattr(selected_model, "bind_tools"):
try:
selected_model.bind_tools(tools)
except Exception as e:
st.warning(f"Failed to bind tools to model: {str(e)}")

# Updated Router: Use LLM to classify intent
def route_agent(state):
if not state["messages"]:
raise ValueError("No messages found in state")
query_text = state['messages'][-1].content
classification_prompt = (
f"You are a query classifier for a multi-agent RAG system. Analyze the user's query and decide the best PRIMARY tool to route to. "
f"Choose based on the dominant intent, even if multiple aspects are mentioned. "
f"If web/online info is needed (e.g., current events, external research), prioritize search_agent, but note subtools for follow-up.\n\n"
f"Available Tools and Keywords/Phrases:\n"
f"- search_agent: For web, search, internet, online, google, latest news, current updates, research outside docs, fetch data, browse, lookup, find info, wikipedia, or any external knowledge not in uploaded PDFs. "
f" (Default if unclear or mixed with others needing web first.)\n"
f"- summarizer: For summarize, summary, overview, condense, in X lines, brief, key highlights, tl;dr, or extract main points from content.\n"
f"- mcq_generator: For mcq, mcqs, multiple choice, questions, quiz, generate questions, practice test, exam questions, or self-quiz.\n"
f"- notes_maker: For notes, make notes, key notes, revision notes, bullet points (but convert to paragraphs), study notes, concise outline, or structured takeaways.\n"
f"- exam_prep_agent: For prepare exam, study plan, revision plan, exam strategy, full prep, timetable, or overall study guide.\n"
f"- concept_explainer: For explain, what is, define, how does it work, breakdown, simple terms, or clarify a concept.\n\n"
f"Query: '{query_text}'\n\n"
f"Respond ONLY with the tool name, e.g., 'search_agent' or 'summarizer'. If the query needs web search first (even with other intents), choose search_agent."
)
try:
classification_response = selected_model.invoke(classification_prompt)
tool_name = getattr(classification_response, "content", str(classification_response)).strip().lower()
# Map to exact tool name
if "search" in tool_name or "web" in tool_name:
tool_name = "search_agent"
elif "summariz" in tool_name:
tool_name = "summarizer"
elif "mcq" in tool_name or "question" in tool_name:
tool_name = "mcq_generator"
elif "notes" in tool_name:
tool_name = "notes_maker"
elif "exam" in tool_name or "prep" in tool_name:
tool_name = "exam_prep_agent"
elif "concept" in tool_name or "explain" in tool_name:
tool_name = "concept_explainer"
else:
tool_name = "search_agent" # Safe default
except Exception as e:
st.warning(f"Classification failed: {e}. Falling back to keyword routing.")
ql = query_text.lower()
if any(word in ql for word in ["web", "search", "internet", "online", "google", "latest", "research", "fetch", "browse", "lookup"]):
tool_name = "search_agent"
elif any(word in ql for word in ["summarize", "summary", "overview", "condense", "brief", "highlights", "tl;dr"]):
tool_name = "summarizer"
elif any(word in ql for word in ["mcq", "mcqs", "multiple choice", "question", "questions", "quiz", "practice", "test"]):
tool_name = "mcq_generator"
elif any(word in ql for word in ["notes", "make notes", "key notes", "revision notes", "study notes", "outline"]):
tool_name = "notes_maker"
elif any(word in ql for word in ["prepare", "exam", "study plan", "revision plan", "strategy", "timetable"]):
tool_name = "exam_prep_agent"
elif any(word in ql for word in ["explain", "what is", "define", "how does", "breakdown", "clarify"]):
tool_name = "concept_explainer"
else:
tool_name = "search_agent"

tool_call = {
"name": tool_name,
"args": {"query": query_text},
"id": f"call_{tool_name}_{hashlib.md5(query_text.lower().encode()).hexdigest()[:8]}"
}
ai_message = AIMessage(content="", tool_calls=[tool_call])
return {"messages": state["messages"] + [ai_message], "next_tool": tool_name, "subtool": "none"}

# Updated Subtool Router
def route_subtool(state):
if state["next_tool"] == "search_agent":
tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.tool_call_id.startswith("call_search_agent")]
if not tool_msgs:
st.error("No ToolMessage found for search_agent. Workflow may have failed.")
return {
"messages": state["messages"] + [ToolMessage(
content="Error: No search_agent output found.",
tool_call_id=f"call_error_{hashlib.md5('search_agent'.encode()).hexdigest()[:8]}"
)],
"subtool": "none"
}

# Extract content from ToolMessage
result_content = tool_msgs[-1].content
if not isinstance(result_content, dict):
try:
# Try to parse as JSON if it's a string
import json
result = json.loads(result_content)
except:
st.error(f"Invalid search_agent output format: expected dict, got {type(result_content)}.")
return {
"messages": state["messages"] + [ToolMessage(
content=f"Error: Invalid search_agent output format. Expected dict, got {type(result_content)}.",
tool_call_id=f"call_error_{hashlib.md5('search_agent'.encode()).hexdigest()[:8]}"
)],
"subtool": "none"
}
else:
result = result_content

subtool = result.get("subtool", "none")
search_content = result.get("content", "")

if not search_content:
st.warning("Search agent returned empty content. Defaulting to 'none' subtool.")
return {
"messages": state["messages"] + [ToolMessage(
content="Error: Search agent returned empty content.",
tool_call_id=f"call_error_{hashlib.md5('search_agent'.encode()).hexdigest()[:8]}"
)],
"subtool": "none"
}

if subtool == "none":
st.info(f"No subtool selected for query. Returning raw search results.")
return {"messages": state["messages"], "subtool": "none"}

# Validate subtool
valid_subtools = ["summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer"]
if subtool not in valid_subtools:
st.error(f"Invalid subtool '{subtool}' selected. Defaulting to raw search results.")
return {
"messages": state["messages"] + [ToolMessage(
content=f"Invalid subtool '{subtool}' selected. Defaulting to raw search results.",
tool_call_id=f"call_error_{hashlib.md5(subtool.encode()).hexdigest()[:8]}"
)],
"subtool": "none"
}

# Create tool call for subtool
query = state["messages"][0].content
tool_call = {
"name": subtool,
"args": {"query": query, "context": search_content},
"id": f"call_{subtool}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
}
ai_message = AIMessage(content="", tool_calls=[tool_call])
st.info(f"Routing to subtool: {subtool} with query: {query}")
return {"messages": state["messages"] + [ai_message], "subtool": subtool}

return {"messages": state["messages"], "subtool": "none"}

# State Definition
class AgentInput(TypedDict):
messages: List[BaseMessage]
next_tool: str
subtool: str

# Build the Graph
graph = StateGraph(AgentInput)
graph.add_node("router", RunnableLambda(route_agent))

# Add nodes for each tool
for tool_func in tools:
graph.add_node(tool_func.name, ToolNode([tool_func]))

graph.add_node("subtool_router", RunnableLambda(route_subtool))

# Conditional edges from router to tools
graph.add_conditional_edges(
"router",
lambda state: state["next_tool"],
{tool.name: tool.name for tool in tools}
)

# Edge from search_agent to subtool_router
graph.add_edge("search_agent", "subtool_router")

# Conditional edges from subtool_router to subtools or END
def get_subtool(state):
subtool = state.get("subtool", "none")
return subtool

graph.add_conditional_edges(
"subtool_router",
get_subtool,
{
"summarizer": "summarizer",
"mcq_generator": "mcq_generator",
"notes_maker": "notes_maker",
"exam_prep_agent": "exam_prep_agent",
"concept_explainer": "concept_explainer",
"none": END
}
)

# Edges from primary tools (except search_agent) to END
for tool_func in tools:
if tool_func.name != "search_agent":
graph.add_edge(tool_func.name, END)

# Edges from subtools to END
subtool_names = ["summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer"]
for subtool_name in subtool_names:
graph.add_edge(subtool_name, END)

graph.set_entry_point("router")
app = graph.compile()

# Streamlit Query Handling
user_input = st.text_input("💬 Ask your query based on the uploaded material or internet:")
if user_input:
try:
state = {"messages": [HumanMessage(content=user_input)], "next_tool": "", "subtool": ""}
output = app.invoke(state)

# Extract the final response
final_response = None
for msg in reversed(output["messages"]):
if isinstance(msg, ToolMessage):
final_response = msg.content
break

st.subheader(f"🧠 Agent Response (Agent: {output.get('next_tool', 'unknown')}, Subtool: {output.get('subtool', 'none')})")

if final_response:
if isinstance(final_response, dict):
st.markdown(final_response.get("content", "No content available"))
else:
st.markdown(final_response)
else:
st.warning("No tool messages found in output. Workflow may have terminated early.")

except Exception as e:
st.error(f"Error processing query: {str(e)}")
import traceback
st.error(f"Traceback: {traceback.format_exc()}")
else:
st.info("👆 Upload PDFs and select a model to enable the agents.")