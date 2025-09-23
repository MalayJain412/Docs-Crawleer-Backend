"""Async web crawler for documentation sites."""

import asyncio
import aiohttp
from typing import List, Set, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime
import time
from collections import deque

try:
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, CrawlSession, LinkInfo
    from crawler.content_parser import ContentParser
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, CrawlSession, LinkInfo
    from crawler.content_parser import ContentParser


class WebCrawler:
    """Async web crawler for documentation sites with domain restriction."""
    
    def __init__(self, custom_session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize the web crawler.
        
        Args:
            custom_session: Optional custom aiohttp session
        """
        self.logger = default_logger
        self.parser = ContentParser()
        self.session = custom_session
        self._own_session = custom_session is None
        
        # Crawling state
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.crawled_documents: List[DocumentContent] = []
        
        # Statistics
        self.start_time: Optional[float] = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_content_length': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=settings.MAX_CONCURRENT_REQUESTS)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'Documentation-Crawler/1.0 (+crawler-bot)'
                }
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session and self._own_session:
            await self.session.close()
    
    async def crawl_domain(self, start_url: str, base_domain: str = None, max_depth: int = None) -> CrawlSession:
        """
        Crawl an entire documentation domain.
        
        Args:
            start_url: Starting URL for crawling
            base_domain: Base domain to restrict crawling (auto-detected if None)
            max_depth: Maximum crawling depth (None for unlimited)
            
        Returns:
            CrawlSession with crawling results
        """
        self.start_time = time.time()
        self.logger.info(f"Starting crawl of {start_url}")
        
        # Auto-detect base domain if not provided
        if not base_domain:
            parsed = urlparse(start_url)
            base_domain = parsed.netloc
        
        # Initialize crawl session
        session = CrawlSession(
            domain=base_domain,
            start_url=start_url,
            domain_folder=URLUtils.extract_domain_name(start_url)
        )
        
        # Reset state
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.crawled_documents.clear()
        
        # Start crawling
        try:
            await self._crawl_recursive(start_url, base_domain, max_depth)
            
            session.status = "completed"
            session.completed_at = time.time()
            
        except Exception as e:
            self.logger.error(f"Crawling failed: {e}")
            session.status = "failed"
            
        # Update session stats
        session.total_pages = len(self.visited_urls) + len(self.failed_urls)
        session.successful_pages = len(self.crawled_documents)
        session.failed_pages = len(self.failed_urls)
        
        crawl_time = time.time() - self.start_time
        self.logger.info(f"Crawling completed in {crawl_time:.2f}s. Success: {session.successful_pages}, Failed: {session.failed_pages}")
        
        return session
    
    async def _crawl_recursive(self, start_url: str, base_domain: str, max_depth: Optional[int]):
        """Recursive crawling implementation using BFS."""
        # Queue of (url, depth) tuples
        crawl_queue = deque([(start_url, 0)])
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        
        while crawl_queue:
            # Get next batch of URLs to process
            current_batch = []
            batch_size = min(settings.MAX_CONCURRENT_REQUESTS, len(crawl_queue))
            
            for _ in range(batch_size):
                if crawl_queue:
                    url, depth = crawl_queue.popleft()
                    
                    # Skip if max depth reached
                    if max_depth is not None and depth > max_depth:
                        continue
                    
                    # Skip if already processed
                    if url in self.visited_urls or url in self.failed_urls:
                        continue
                    
                    current_batch.append((url, depth))
            
            if not current_batch:
                break
            
            # Process batch concurrently
            tasks = [
                self._process_url(semaphore, url, depth, base_domain, crawl_queue)
                for url, depth in current_batch
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add delay between batches
            if crawl_queue:
                await asyncio.sleep(settings.DELAY_BETWEEN_REQUESTS)
    
    async def _process_url(self, semaphore: asyncio.Semaphore, url: str, depth: int, base_domain: str, crawl_queue: deque):
        """Process a single URL with retry logic."""
        async with semaphore:
            for attempt in range(settings.RETRY_ATTEMPTS):
                try:
                    document = await self._fetch_and_parse(url, base_domain)
                    
                    if document:
                        self.crawled_documents.append(document)
                        self.visited_urls.add(url)
                        self.stats['successful_requests'] += 1
                        self.stats['total_content_length'] += document.content_length
                        
                        # Add internal links to queue for next depth level
                        for link in document.links:
                            if (link.is_internal and 
                                link.url not in self.visited_urls and 
                                link.url not in self.failed_urls and
                                URLUtils.is_valid_url(link.url)):
                                crawl_queue.append((link.url, depth + 1))
                        
                        self.logger.info(f"Successfully crawled {url} (depth {depth}) - {document.content_length} chars")
                        break
                    else:
                        raise Exception("Failed to parse content")
                        
                except Exception as e:
                    if attempt == settings.RETRY_ATTEMPTS - 1:
                        # Final attempt failed
                        self.failed_urls.add(url)
                        self.stats['failed_requests'] += 1
                        self.logger.warning(f"Failed to crawl {url} after {settings.RETRY_ATTEMPTS} attempts: {e}")
                    else:
                        # Wait before retry
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _fetch_and_parse(self, url: str, base_domain: str) -> Optional[DocumentContent]:
        """Fetch URL content and parse it."""
        try:
            self.stats['total_requests'] += 1
            
            async with self.session.get(url) as response:
                # Check if response is successful
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                # Check content type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    raise Exception(f"Not HTML content: {content_type}")
                
                # Read content
                html_content = await response.text(encoding='utf-8', errors='ignore')
                
                # Parse content
                document = self.parser.parse_content(html_content, url, base_domain)
                
                return document
                
        except asyncio.TimeoutError:
            raise Exception("Request timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error: {e}")
        except Exception as e:
            raise Exception(f"Fetch error: {e}")
    
    async def crawl_single_page(self, url: str, base_domain: str = None) -> Optional[DocumentContent]:
        """
        Crawl a single page without following links.
        
        Args:
            url: URL to crawl
            base_domain: Base domain for link classification
            
        Returns:
            DocumentContent or None if failed
        """
        if not base_domain:
            parsed = urlparse(url)
            base_domain = parsed.netloc
        
        try:
            document = await self._fetch_and_parse(url, base_domain)
            if document:
                self.logger.info(f"Successfully crawled single page: {url}")
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to crawl single page {url}: {e}")
            return None
    
    def get_crawl_statistics(self) -> Dict[str, Any]:
        """Get current crawling statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100,
            'total_documents': len(self.crawled_documents),
            'total_content_length': self.stats['total_content_length'],
            'average_content_length': self.stats['total_content_length'] / max(len(self.crawled_documents), 1),
            'elapsed_time': elapsed_time,
            'pages_per_second': len(self.crawled_documents) / max(elapsed_time, 1)
        }
    
    def get_internal_links(self) -> List[str]:
        """Get all unique internal links found during crawling."""
        internal_links = set()
        
        for doc in self.crawled_documents:
            for link in doc.links:
                if link.is_internal:
                    internal_links.add(link.url)
        
        return sorted(list(internal_links))
    
    def get_external_links(self) -> List[str]:
        """Get all unique external links found during crawling."""
        external_links = set()
        
        for doc in self.crawled_documents:
            for link in doc.links:
                if not link.is_internal:
                    external_links.add(link.url)
        
        return sorted(list(external_links))