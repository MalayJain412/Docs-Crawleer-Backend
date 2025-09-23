"""Content parser using trafilatura for extracting clean text from web pages."""

import trafilatura
from trafilatura.settings import use_config
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, urlparse
import re

try:
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, LinkInfo
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.logger import default_logger
    from utils.url_utils import URLUtils
    from storage.schemas import DocumentContent, LinkInfo


class ContentParser:
    """Handles content extraction and parsing from HTML."""
    
    def __init__(self):
        """Initialize the content parser."""
        self.logger = default_logger
        
        # Configure trafilatura for better extraction
        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
        self.config.set("DEFAULT", "MIN_OUTPUT_SIZE", "50")
        
    def parse_content(self, html_content: str, url: str, base_domain: str) -> Optional[DocumentContent]:
        """
        Parse HTML content and extract structured information.
        
        Args:
            html_content: Raw HTML content
            url: URL of the page
            base_domain: Base domain for link classification
            
        Returns:
            DocumentContent object or None if parsing fails
        """
        try:
            # Extract main content using trafilatura
            extracted_text = trafilatura.extract(
                html_content,
                config=self.config,
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_images=False
            )
            
            if not extracted_text or len(extracted_text.strip()) < 50:
                self.logger.warning(f"Insufficient content extracted from {url}")
                return None
            
            # Extract title
            title = self._extract_title(html_content, extracted_text)
            
            # Extract metadata
            metadata = self._extract_metadata(html_content, url)
            
            # Extract and classify links
            links = self._extract_and_classify_links(html_content, url, base_domain)
            
            # Clean and process content
            cleaned_content = self._clean_content(extracted_text)
            
            # Detect language
            language = self._detect_language(cleaned_content)
            
            return DocumentContent(
                url=url,
                title=title,
                content=cleaned_content,
                raw_html=html_content if len(html_content) < 100000 else None,  # Store raw HTML only if not too large
                metadata=metadata,
                links=links,
                content_length=len(cleaned_content),
                language=language
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing content from {url}: {e}")
            return None
    
    def _extract_title(self, html_content: str, extracted_text: str) -> Optional[str]:
        """Extract page title from HTML or content."""
        try:
            # First try to get title from HTML
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                # Clean common patterns
                title = re.sub(r' - [^-]+$', '', title)  # Remove site name suffix
                title = re.sub(r' \| [^|]+$', '', title)  # Remove pipe separator suffix
                if title:
                    return title
            
            # Fallback: use first line of extracted text if it looks like a heading
            first_lines = extracted_text.split('\n')[:3]
            for line in first_lines:
                line = line.strip()
                if line and len(line) < 200 and not line.endswith('.'):
                    return line
            
            return None
            
        except Exception:
            return None
    
    def _extract_metadata(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc
        }
        
        try:
            # Extract meta description
            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if desc_match:
                metadata['description'] = desc_match.group(1).strip()
            
            # Extract meta keywords
            keywords_match = re.search(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if keywords_match:
                metadata['keywords'] = [kw.strip() for kw in keywords_match.group(1).split(',')]
            
            # Extract canonical URL
            canonical_match = re.search(r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if canonical_match:
                metadata['canonical_url'] = canonical_match.group(1)
            
            # Extract Open Graph data
            og_title = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if og_title:
                metadata['og_title'] = og_title.group(1)
            
            og_desc = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            if og_desc:
                metadata['og_description'] = og_desc.group(1)
            
        except Exception as e:
            self.logger.debug(f"Error extracting metadata from {url}: {e}")
        
        return metadata
    
    def _extract_and_classify_links(self, html_content: str, base_url: str, base_domain: str) -> List[LinkInfo]:
        """Extract and classify links from HTML content."""
        links = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            seen_urls = set()
            
            for link_elem in soup.find_all('a', href=True):
                href = link_elem.get('href', '').strip()
                if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                    continue
                
                # Get link text
                link_text = link_elem.get_text(strip=True)
                
                # Resolve relative URLs
                absolute_url = URLUtils.resolve_relative_url(base_url, href)
                normalized_url = URLUtils.normalize_url(absolute_url)
                
                # Skip duplicates
                if normalized_url in seen_urls:
                    continue
                seen_urls.add(normalized_url)
                
                # Skip invalid URLs
                if not URLUtils.is_valid_url(normalized_url):
                    continue
                
                # Classify as internal or external
                is_internal = URLUtils.is_docs_url(normalized_url, base_domain)
                
                links.append(LinkInfo(
                    url=normalized_url,
                    text=link_text[:200] if link_text else None,  # Limit text length
                    is_internal=is_internal,
                    is_visited=False
                ))
            
        except Exception as e:
            self.logger.debug(f"Error extracting links from {base_url}: {e}")
        
        return links
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""
        
        try:
            # Remove excessive whitespace
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            # Remove common navigation patterns
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip common navigation/footer patterns
                if any(pattern in line.lower() for pattern in [
                    'skip to main content',
                    'copyright ©',
                    'all rights reserved',
                    'privacy policy',
                    'terms of service',
                    'cookie policy'
                ]):
                    continue
                
                # Skip very short lines that are likely navigation
                if len(line) < 3:
                    continue
                
                cleaned_lines.append(line)
            
            content = '\n'.join(cleaned_lines)
            
            # Final cleanup
            content = content.strip()
            
            return content
            
        except Exception as e:
            self.logger.debug(f"Error cleaning content: {e}")
            return content
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Detect content language (simple heuristic)."""
        if not content:
            return None
        
        try:
            # Simple heuristic based on common English words
            english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
            words = content.lower().split()[:100]  # Check first 100 words
            
            english_count = sum(1 for word in words if word in english_indicators)
            
            if english_count > len(words) * 0.1:  # If >10% are common English words
                return 'en'
            
            return 'unknown'
            
        except Exception:
            return None
    
    def extract_text_only(self, html_content: str) -> Optional[str]:
        """
        Extract only clean text content without metadata.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Clean text content or None
        """
        try:
            extracted = trafilatura.extract(
                html_content,
                config=self.config,
                include_comments=False,
                include_tables=True,
                include_links=False,
                include_images=False
            )
            
            return self._clean_content(extracted) if extracted else None
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return None