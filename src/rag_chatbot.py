from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI  # Changed from OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
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
        """Load and prepare documents.

        The JSON payload may not include an ``abstract`` key, so we
        gracefully fall back to an empty string and still build a
        document from the other available metadata (title, authors,
        year, publication, url, etc.).  All of the metadata is preserved
        on the ``Document`` so it can later be inspected if needed.
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            title = item.get('title', 'Untitled')
            abstract = item.get('abstract', '')  # may be missing
            authors = item.get('authors', '')
            year = item.get('year', '')
            publication = item.get('publication', '')

            # build a readable content string from whatever we have
            parts = [f"Title: {title}"]
            if authors:
                parts.append(f"Authors: {authors}")
            if year:
                parts.append(f"Year: {year}")
            if publication:
                parts.append(f"Publication: {publication}")
            if abstract:
                parts.append(f"\nAbstract: {abstract}")

            content = "\n".join(parts)

            metadata = {
                'title': title,
                'url': item.get('url', ''),
                'authors': authors,
                'year': year,
                'publication': publication,
            }

            doc = Document(
                page_content=content,
                metadata=metadata
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
        llm = ChatOpenAI(  # Changed from OpenAI to ChatOpenAI
            base_url=self.lm_studio_url,
            api_key="not-needed",
            temperature=0.7,
            model="local-model"  # LM Studio requires a model name
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
        
        result = self.qa_chain.invoke({"query": question})  # Changed to .invoke()
        return {
            'answer': result['result'],
            'sources': [doc.metadata for doc in result['source_documents']]
        }
    