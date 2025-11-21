from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- THE FIX: Document is now in langchain_core ---
from langchain_core.documents import Document 
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

class RAGChatbot:
    def __init__(self):
        print("Initializing RAG Vector Store...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
        # Detect Device
        device = -1
        if torch.backends.mps.is_available():
            device = "mps"
        
        print(f"Loading LLM on {device}...")
        # Using Flan-T5-Large
        pipe = pipeline(
            "text2text-generation", 
            model="google/flan-t5-large", 
            device=device, 
            max_length=512
        ) 
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def ingest_insights(self, insights):
        docs = [Document(page_content=text) for text in insights]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        print("Vector Store Created.")

    def ask(self, query):
        if not self.vector_store:
            return "Please process the video first.", ""
        
        # Retrieve relevant captions
        docs = self.vector_store.similarity_search(query, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        # Debugging logs
        print(f"\n[DEBUG] Question: {query}")
        print(f"[DEBUG] Retrieved Context: {context[:200]}...") 
        
        # Optimized Prompt
        prompt = f"""
        Answer the question using only the context provided below.
        
        Question: {query}
        
        Context:
        {context}
        
        Answer:"""
        
        response = self.llm.invoke(prompt)
        return response, context
