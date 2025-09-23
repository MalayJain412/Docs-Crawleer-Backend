"""URL utilities for normalization and validation."""

import re
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Optional, Set


class URLUtils:
    """Utility class for URL operations."""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL by removing anchors and unnecessary query parameters.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
        """
        parsed = urlparse(url)
        # Remove fragment (anchor)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,  # Keep query params for now
            ''  # Remove fragment
        ))
        
        # Remove trailing slash for consistency
        if normalized.endswith('/') and len(normalized) > 1:
            normalized = normalized[:-1]
            
        return normalized
    
    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        """
        Check if two URLs belong to the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if URLs are from the same domain
        """
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()
            return domain1 == domain2
        except Exception:
            return False
    
    @staticmethod
    def is_docs_url(url: str, base_domain: str) -> bool:
        """
        Check if URL belongs to the docs domain and should be crawled.
        
        Args:
            url: URL to check
            base_domain: Base domain to match against
            
        Returns:
            True if URL should be crawled
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Check if domain matches and path suggests documentation
            return (
                base_domain.lower() in domain and
                ('doc' in path or 'guide' in path or 'tutorial' in path or path.startswith('/') or 'api' in path)
            )
        except Exception:
            return False
    
    @staticmethod
    def extract_domain_name(url: str) -> str:
        """
        Extract a clean domain name for folder naming.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Clean domain name suitable for folder names
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove common prefixes
            domain = re.sub(r'^(www\.|docs\.|api\.)', '', domain)
            
            # Replace dots and special chars with dashes
            domain = re.sub(r'[^\w-]', '-', domain)
            
            # Remove multiple consecutive dashes
            domain = re.sub(r'-+', '-', domain)
            
            # Remove leading/trailing dashes
            domain = domain.strip('-')
            
            return domain or "unknown-domain"
        except Exception:
            return "unknown-domain"
    
    @staticmethod
    def resolve_relative_url(base_url: str, relative_url: str) -> str:
        """
        Resolve a relative URL against a base URL.
        
        Args:
            base_url: Base URL
            relative_url: Relative URL
            
        Returns:
            Absolute URL
        """
        try:
            return urljoin(base_url, relative_url)
        except Exception:
            return relative_url
    
    @staticmethod
    def extract_links_from_html(html_content: str, base_url: str) -> List[str]:
        """
        Extract all links from HTML content.
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href:
                    absolute_url = URLUtils.resolve_relative_url(base_url, href)
                    normalized_url = URLUtils.normalize_url(absolute_url)
                    if normalized_url not in links:
                        links.append(normalized_url)
            
            return links
        except Exception:
            return []
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is valid and crawlable.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid
        """
        try:
            parsed = urlparse(url)
            return all([
                parsed.scheme in ['http', 'https'],
                parsed.netloc,
                not url.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.dmg'))
            ])
        except Exception:
            return False