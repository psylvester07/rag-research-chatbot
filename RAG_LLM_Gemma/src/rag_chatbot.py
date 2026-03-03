from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import json

class RAGChatbot:
    def __init__(self, lm_studio_url="http://localhost:1234/v1"):
        self.lm_studio_url = lm_studio_url
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self, data_path='data/raw/research_data.json'):
        """Load and prepare documents"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            content = f"Title: {item['title']}\n\nAbstract: {item['abstract']}"
            doc = Document(
                page_content=content,
                metadata={'title': item['title'], 'url': item.get('url', '')}
            )
            documents.append(doc)
        
        return documents
    
    def create_vectorstore(self, documents):
        """Split documents and create vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="data/processed/chroma_db"
        )
    
    def setup_qa_chain(self):
        """Setup QA chain with LM Studio"""
        from langchain_community.llms import OpenAI
        
        llm = OpenAI(
            base_url=self.lm_studio_url,
            api_key="not-needed",  # LM Studio doesn't require API key
            temperature=0.7
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
    
    def query(self, question):
        """Ask a question to the chatbot"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run setup_qa_chain first.")
        
        result = self.qa_chain({"query": question})
        return {
            'answer': result['result'],
            'sources': [doc.metadata for doc in result['source_documents']]
        }
