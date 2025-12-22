from data_collector import ResearchDataCollector
from rag_chatbot import RAGChatbot
import argparse

def main():
    # Step 1: Collect data
    print("Collecting research data...")
    collector = ResearchDataCollector()
    collector.fetch_arxiv_papers(query="machine learning", max_results=50)
    collector.save_data()
    print(f"Collected {len(collector.data)} papers")
    
    # Step 2: Initialize RAG chatbot
    print("\nInitializing RAG chatbot...")
    chatbot = RAGChatbot()
    
    # Step 3: Load and process documents
    print("Loading documents...")
    documents = chatbot.load_documents()
    
    print("Creating vector store...")
    chatbot.create_vectorstore(documents)
    
    print("Setting up QA chain...")
    chatbot.setup_qa_chain()
    
    # Step 4: Interactive chat
    print("\n=== RAG Chatbot Ready ===")
    print("Ask questions about the research papers (type 'quit' to exit)\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit']:
            break
        
        result = chatbot.query(question)
        print(f"\nBot: {result['answer']}\n")
        print("Sources:")
        for source in result['sources']:
            print(f"  - {source['title']}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect-data', action='store_true', 
                       help='Fetch fresh data before running')
    args = parser.parse_args()
    
    if args.collect_data:
        from data_collector import ResearchDataCollector
        collector = ResearchDataCollector()
        collector.fetch_nuailab_publications()
        collector.save_data()
    
    # Then run normal chatbot flow
    main()
