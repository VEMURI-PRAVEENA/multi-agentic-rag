# 📚 Multi-Agentic RAG - College Study Assistant

A Streamlit-based multi-agent RAG (Retrieval-Augmented Generation) system for students.  
It helps with summarization, MCQ generation, note-making, exam preparation, and concept explanations using uploaded PDFs or live web search.

---

## 🚀 Features

### **🔹 Multiple LLMs Supported**
- Gemini 2.5 Flash  
- Gemini 2.5 Flash Lite  
- Gemini 2.0 Flash  
- GPT-OSS-120B (via Groq)

---

### **🔹 Document Handling**
- Upload multiple PDFs  
- FAISS vector store with caching (hash-based reuse)  
- Automatic text chunking & embedding generation  

---

### **🔹 Study Agents**
- 📑 **Summarizer** – Summarizes documents in custom lengths  
- ❓ **MCQ Generator** – Creates exam-style MCQs (supports ASCII diagrams)  
- 📝 **Notes Maker** – Creates concise, structured notes  
- 📅 **Exam Prep Agent** – Generates study plans and revision tips  
- 💡 **Concept Explainer** – Explains topics in simple terms  
- 🔍 **Search Agent** – Uses Google Custom Search for web knowledge  

---

### **🔹 Dynamic Routing**
- Auto-detects user query intent  
- Routes to correct tool (summarizer, MCQ generator, notes, etc.)  
- Web-related queries auto-processed via Search Agent  
- Sub-routing for summarizing web pages, MCQ from web, notes from web, etc.

---

## 🏗 Project Architecture (Visual Flow)
<img width="784" height="724" alt="image" src="https://github.com/user-attachments/assets/07f26e0b-2aaf-4ce3-8c05-75da3a589f40" />


## ▶️ Usage

### Run the Streamlit app:
```bash
streamlit run multiagenticRag.py
```

### 1️⃣ Upload your study material (PDFs)

### 2️⃣ Select your preferred language model

### 3️⃣ Ask queries like:
- "Summarize chapter 3 in 10 lines"
- "Generate 20 MCQs for Thermodynamics"
- "Make notes on Electrochemistry in 15 lines"
- "Explain Ohm's Law in 5 lines"
- "Prepare a study plan for Organic Chemistry"

### 4️⃣ The system will automatically:
- Detect the type of query  
- Route it to the correct agent (summarizer, MCQ generator, notes maker, etc.)  
- Produce accurate, structured study material
