import requests
import json
import time
from typing import List, Dict

class ResearchDataCollector:
    def __init__(self, output_path='data/raw/research_data.json'):
        self.output_path = output_path
        self.data = []
    
    def fetch_nuailab_publications(self, url='https://nuailab.com/data/publications.json'):
        """
        Fetch research publications from NuAILab JSON endpoint
        
        Args:
            url: URL to the publications JSON file
        """
        try:
            print(f"Fetching publications from {url}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            json_data = response.json()
            publications = json_data.get('publications', [])
            
            print(f"Found {len(publications)} publications")
            
            # Transform the data to match our format
            for pub in publications:
                # Skip entries with placeholder data
                if pub.get('title') == 'title' or pub.get('authors') == 'author':
                    continue
                    
                # Create a structured entry
                entry = {
                    'title': pub.get('title', ''),
                    'authors': pub.get('authors', ''),
                    'abstract': f"Publication: {pub.get('pub', '')}. Year: {pub.get('year', '')}",
                    'publication': pub.get('pub', ''),
                    'year': pub.get('year', ''),
                    'url': pub.get('link', ''),
                    'source': 'NuAILab'
                }
                
                self.data.append(entry)
            
            print(f"Successfully processed {len(self.data)} publications")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    def fetch_arxiv_papers(self, query: str, max_results: int = 50):
        """
        Fetch papers from arXiv (optional - for additional data sources)
        
        Args:
            query: Search query for arXiv
            max_results: Maximum number of results to fetch
        """
        try:
            import arxiv
            
            print(f"Fetching arXiv papers for query: '{query}'...")
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in search.results():
                entry = {
                    'title': result.title,
                    'authors': ', '.join([author.name for author in result.authors]),
                    'abstract': result.summary,
                    'publication': 'arXiv',
                    'year': str(result.published.year),
                    'url': result.entry_id,
                    'source': 'arXiv'
                }
                self.data.append(entry)
            
            print(f"Successfully fetched {len([d for d in self.data if d['source'] == 'arXiv'])} arXiv papers")
            
        except ImportError:
            print("arxiv library not installed. Run: pip install arxiv")
        except Exception as e:
            print(f"Error fetching arXiv data: {e}")
    
    def scrape_website(self, url: str, selector: str = None):
        """
        Generic web scraper for research abstracts
        
        Args:
            url: URL to scrape
            selector: CSS selector for abstract elements (optional)
        """
        try:
            from bs4 import BeautifulSoup
            
            print(f"Scraping website: {url}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Customize this based on your target website structure
            if selector:
                abstracts = soup.select(selector)
            else:
                # Default: look for common abstract containers
                abstracts = soup.find_all(['div', 'p'], class_=['abstract', 'summary'])
            
            for idx, abstract in enumerate(abstracts):
                entry = {
                    'title': f"Document {idx + 1} from {url}",
                    'authors': 'Unknown',
                    'abstract': abstract.get_text(strip=True),
                    'publication': 'Web Scraped',
                    'year': 'N/A',
                    'url': url,
                    'source': 'Web Scraping'
                }
                self.data.append(entry)
            
            print(f"Scraped {len(abstracts)} items from website")
            
        except ImportError:
            print("beautifulsoup4 library not installed. Run: pip install beautifulsoup4")
        except Exception as e:
            print(f"Error scraping website: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        if not self.data:
            return {"total": 0}
        
        sources = {}
        years = {}
        
        for entry in self.data:
            source = entry.get('source', 'Unknown')
            year = entry.get('year', 'Unknown')
            
            sources[source] = sources.get(source, 0) + 1
            years[year] = years.get(year, 0) + 1
        
        return {
            'total': len(self.data),
            'by_source': sources,
            'by_year': years
        }
    
    def save_data(self):
        """Save collected data to JSON file"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        print(f"\nData saved to: {self.output_path}")
        
        # Print statistics
        stats = self.get_statistics()
        print(f"\n=== Collection Statistics ===")
        print(f"Total papers: {stats['total']}")
        print(f"\nBy source:")
        for source, count in stats.get('by_source', {}).items():
            print(f"  {source}: {count}")
    
    def load_existing_data(self):
        """Load previously saved data"""
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} existing entries")
        except FileNotFoundError:
            print("No existing data file found")
        except Exception as e:
            print(f"Error loading data: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = ResearchDataCollector()
    
    # Fetch NuAILab publications
    collector.fetch_nuailab_publications()
    
    # Optional: Add more data from arXiv
    # collector.fetch_arxiv_papers(query="neuromorphic computing", max_results=20)
    
    # Optional: Scrape additional websites
    # collector.scrape_website("https://example.com/research", selector=".abstract")
    
    # Save all collected data
    collector.save_data()
