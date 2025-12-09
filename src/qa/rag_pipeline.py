"""Retrieval-Augmented Generation (RAG) pipeline for Q/A system."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import QueryRequest, QueryResponse
    from embeddings.embedding_service import EmbeddingService
    from embeddings.vector_store import VectorStore
    from embeddings.multi_domain_vector_store import MultiDomainVectorStore, DomainSearchResult
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config.settings import settings
    from utils.logger import default_logger
    from storage.schemas import QueryRequest, QueryResponse
    from embeddings.embedding_service import EmbeddingService
    from embeddings.vector_store import VectorStore
    from embeddings.multi_domain_vector_store import MultiDomainVectorStore, DomainSearchResult


class RAGPipeline:
    """RAG pipeline for answering queries using retrieved document chunks."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.logger = default_logger
        self.embedding_service = EmbeddingService()
        self.vector_stores: Dict[str, VectorStore] = {}
        self.multi_domain_store = MultiDomainVectorStore(max_concurrent_domains=5)
        self._llm_client = None
        
    async def initialize(self):
        """Initialize the RAG pipeline components."""
        try:
            # Initialize embedding service
            await self.embedding_service.initialize()
            
            # Initialize LLM client
            self._llm_client = await self._init_llm_client()
            
            # Initialize multi-domain store semaphore (ensure it's created in correct event loop)
            self.multi_domain_store = MultiDomainVectorStore(max_concurrent_domains=5)
            
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def _init_llm_client(self) -> Optional[Any]:
        """Initialize LLM client (Gemini or fallback)."""
        try:
            if settings.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                
                # Test the model with Gemini Flash
                model = genai.GenerativeModel(settings.GEMINI_LLM_MODEL)
                test_response = model.generate_content("Hello")
                
                self.logger.info("Initialized Gemini LLM client")
                return genai
            else:
                self.logger.warning("No LLM API key provided, running in embedding-only mode")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.logger.info("Running in embedding-only mode (no LLM available)")
            return None
    
    def add_vector_store(self, domain: str, vector_store: VectorStore):
        """
        Add a vector store for a domain.
        
        Args:
            domain: Domain name
            vector_store: VectorStore instance
        """
        self.vector_stores[domain] = vector_store
        self.logger.info(f"Added vector store for domain: {domain}")
    
    def load_vector_store(self, domain: str, domain_folder: str) -> bool:
        """
        Load vector store for a domain.
        
        Args:
            domain: Domain name
            domain_folder: Path to domain folder
            
        Returns:
            True if loaded successfully
        """
        try:
            vector_store = VectorStore(domain, domain_folder)
            if vector_store.load_index():
                self.vector_stores[domain] = vector_store
                self.logger.info(f"Loaded vector store for domain: {domain}")
                return True
            else:
                self.logger.warning(f"No index found for domain: {domain}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load vector store for domain {domain}: {e}")
            return False
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query using RAG pipeline.
        
        Args:
            request: QueryRequest object
            
        Returns:
            QueryResponse with answer and sources
        """
        start_time = time.time()
        
        try:
            # Validate domain
            if request.domain not in self.vector_stores:
                raise ValueError(f"Domain '{request.domain}' not available")
            
            vector_store = self.vector_stores[request.domain]
            
            # Generate query embedding
            query_embeddings = await self.embedding_service.generate_embeddings([request.query])
            if not query_embeddings:
                raise Exception("Failed to generate query embedding")
            
            query_embedding = query_embeddings[0]
            
            # Retrieve relevant chunks
            search_results = vector_store.search(query_embedding, request.top_k)
            
            if not search_results:
                return QueryResponse(
                    query=request.query,
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Generate answer using LLM
            if self._llm_client and request.include_context:
                answer = await self._generate_llm_answer(request.query, search_results)
                confidence = self._calculate_confidence(search_results)
            else:
                # Fallback: return most relevant chunk content
                answer = self._create_fallback_answer(search_results)
                confidence = search_results[0]['similarity_score'] if search_results else 0.0
            
            # Prepare sources
            sources = self._prepare_sources(search_results)
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return QueryResponse(
                query=request.query,
                answer=f"Sorry, I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def _generate_llm_answer(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive answer using LLM with retrieved context."""
        try:
            import google.generativeai as genai
            
            # Prepare context from search results
            context_parts = []
            max_chunks = min(settings.RAG_CONTEXT_CHUNKS, len(search_results))
            
            for i, result in enumerate(search_results[:max_chunks]):
                content = result.get('content', '')[:settings.RAG_CONTENT_LENGTH]
                source_url = result.get('source_url', 'Unknown')
                title = result.get('metadata', {}).get('title', 'Untitled')
                
                context_parts.append(f"[Source {i+1}: {title} - {source_url}]\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Enhanced comprehensive prompt template for markdown output
            prompt = f"""You are an expert technical assistant (senior engineer / architect) whose job is to produce **complete, actionable, copy-pasteable** technical guidance based **only** on the Documentation Context provided in the `context` variable and the user's `query`. Your audience is an engineer who wants runnable steps, not vague summaries.

**IMPORTANT: Your response must be in valid Markdown format that will be parsed and displayed in a web browser.**

======================================================================
NEW RULE — Documentation vs Suggestions:
- For each major section, explicitly separate:
  1. **Logic from Documentation** → Facts, instructions, or explanations found in `context` (with citations).
  2. **Next-step Suggestions** → Actionable recommendations, extensions, or optimizations *based on the documentation* but not literally in it.
- Mark clearly using headings:
  - `### Logic from Documentation`
  - `### Next-step Suggestions`

======================================================================
RULES — Retrieval & Context:
1. Use ONLY the provided `context`. Treat it as ground truth.
2. Always cite inline like this: (Source: [Title]) after any claim.
3. If a query cannot be answered directly, provide:
   - A list of the closest relevant topics or sections (with citations).
   - Explicitly state: "**No direct answer found. Here are related areas you may explore.**"
4. Prioritize **page-level context** and keep related chunks grouped together.
5. Do not hallucinate APIs, parameters, or commands. If something is missing, say "**ASSUMPTION:**" and explain.
6. If retrieval confidence < 0.5: Return related **topics with their doc titles and URLs**, instead of making up an answer.
7. If retrieval yields nothing: Say "**No relevant context retrieved.** Please refine your query or check documentation coverage."

======================================================================
RULES — Response Structure (for engineers):
Always follow this section order for setup/how-to style queries:

- `# TL;DR` — 1–3 sentence summary.
- `# Overview`  
   - `### Logic from Documentation`  
   - `### Next-step Suggestions`
- `# Prerequisites`
   - `### Logic from Documentation`
   - `### Next-step Suggestions`
- `# Architecture`
   - `### Logic from Documentation`
   - `### Next-step Suggestions`
- `# Step-by-step Implementation`
   - `### Logic from Documentation`
   - `### Next-step Suggestions`
- `# Example (Minimal, runnable)`
- `# Configuration & Secrets`
- `# Advanced / Production Notes`
- `# Troubleshooting`
- `# Next steps / Further improvements`
- `# Checklist`
- `# Copy & Paste`
- `# Verification`
- `# Sources`

======================================================================
RULES — Code & Markdown:
- Use fenced code blocks with language identifiers (`bash`, `python`, `yaml`, etc.).
- Use `inline code` for commands, filenames, and variables.
- Provide **complete files** (not fragments) for any file you tell the user to create.
- For every runnable example, include exact installation and run commands.
- Use **bold** for emphasis, *italic* for secondary emphasis, and blockquotes for notes.
- Max 3 emojis for readability.

======================================================================
RULES — Multi-domain Context:
1. Always attribute facts to their originating document, e.g., (Source: [Kubernetes Docs]).
2. If multiple domains are relevant, group content under a new section:
   - `# Domain-specific Notes`
     - `## [Domain/Doc Title]`
       - `### Logic from Documentation`
       - `### Next-step Suggestions`
3. If instructions from different domains conflict, list both sets with their sources, then recommend a safe default under "Next-step Suggestions".

======================================================================
Context from Documentation:
{context}

Query:
{query}

======================================================================
**OUTPUT REQUIREMENTS:**
- Respond ONLY in valid Markdown format
- Explicitly separate documentation facts and suggestions under the headings above
- Ensure all code blocks have proper language tags
- Make the response copy-pasteable and immediately actionable
- End with both `# Copy & Paste` and `# Verification` sections
- If no answer possible, fallback to related topics mode as per Retrieval Rules.

---
**Comprehensive Technical Answer:**
"""
            
            # Generate response with Gemini Flash - configured for comprehensive responses
            model = genai.GenerativeModel(
                settings.GEMINI_LLM_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.LLM_TEMPERATURE,
                    max_output_tokens=settings.LLM_MAX_TOKENS,
                    top_p=0.8,
                    top_k=40
                )
            )
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return self._create_fallback_answer(search_results)
                
        except Exception as e:
            self.logger.warning(f"LLM answer generation failed: {e}")
            return self._create_fallback_answer(search_results)
    
    def _create_fallback_answer(self, search_results: List[Dict[str, Any]]) -> str:
        """Create comprehensive fallback answer from search results when LLM is unavailable."""
        if not search_results:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your query or check if the domain has been crawled and indexed."
        
        # Analyze the query and results to determine response type
        top_result = search_results[0]
        query_lower = ""  # Will be set from context if available
        
        # Extract more content for comprehensive response
        all_content = []
        all_titles = []
        unique_sources = set()
        
        for result in search_results[:5]:  # Use top 5 results
            content = result.get('content', '').strip()
            title = result.get('metadata', {}).get('title', 'Documentation')
            url = result.get('source_url', '')
            
            if content and len(content) > 50:  # Skip very short content
                all_content.append(content)
                all_titles.append(title)
                if url:
                    unique_sources.add(f"({title})")
        
        if not all_content:
            return "No relevant content found in the documentation."
        
        # Create comprehensive structured response
        response_parts = []
        
        # TL;DR section
        response_parts.append("# TL;DR")
        main_topic = self._extract_main_topic(all_content[0])
        response_parts.append(f"Based on the available documentation, here's what you need to know about {main_topic}. The information below is compiled from {len(unique_sources)} documentation sources.")
        response_parts.append("")
        
        # Overview section
        response_parts.append("# Overview")
        overview = self._create_overview_from_content(all_content[:2])
        response_parts.append(overview)
        response_parts.append("")
        
        # Main content sections
        response_parts.append("# Key Information")
        for i, (content, title) in enumerate(zip(all_content[:3], all_titles[:3]), 1):
            response_parts.append(f"## {i}. From {title}")
            
            # Clean and structure the content
            structured_content = self._structure_content(content)
            response_parts.append(structured_content)
            response_parts.append("")
        
        # Add implementation steps if content suggests it
        if self._contains_implementation_info(all_content):
            response_parts.append("# Implementation Steps")
            steps = self._extract_implementation_steps(all_content)
            for i, step in enumerate(steps, 1):
                response_parts.append(f"{i}. {step}")
            response_parts.append("")
        
        # Add code examples if found
        code_examples = self._extract_code_examples(all_content)
        if code_examples:
            response_parts.append("# Code Examples")
            for example in code_examples:
                response_parts.append(f"```{example['language']}")
                response_parts.append(example['code'])
                response_parts.append("```")
                response_parts.append("")
        
        # Next steps section
        response_parts.append("# Next Steps")
        next_steps = self._generate_next_steps(all_content, unique_sources)
        response_parts.append(next_steps)
        response_parts.append("")
        
        # Sources section
        response_parts.append("# Sources")
        for result in search_results[:5]:
            title = result.get('metadata', {}).get('title', 'Untitled')
            url = result.get('source_url', 'No URL')
            score = result.get('similarity_score', 0.0)
            response_parts.append(f"- **{title}** - Relevance: {score:.2f}")
            if url and url != 'No URL':
                response_parts.append(f"  - {url}")
        
        # Confidence indicator
        top_score = search_results[0].get('similarity_score', 0.0)
        if top_score > 0.8:
            confidence_note = "\n---\n✅ **High confidence** - Strong match found in documentation"
        elif top_score > 0.6:
            confidence_note = "\n---\n⚠️ **Moderate confidence** - Good match found, but consider refining your query"
        else:
            confidence_note = "\n---\n❓ **Lower confidence** - Limited matches found, try rephrasing your question"
        
        response_parts.append(confidence_note)
        
        return "\n".join(response_parts)
    
    def _extract_main_topic(self, content: str) -> str:
        """Extract the main topic from content."""
        # Simple heuristic to extract main topic
        words = content.lower().split()
        common_terms = ['agent', 'api', 'setup', 'configuration', 'installation', 'quickstart', 'guide']
        
        for term in common_terms:
            if term in words:
                return term
        
        # Fallback to first few words
        return ' '.join(content.split()[:3])
    
    def _create_overview_from_content(self, contents: List[str]) -> str:
        """Create overview from multiple content pieces."""
        combined_text = " ".join(contents[:2])[:500]  # First 500 chars from top 2 results
        
        # Extract key phrases
        sentences = combined_text.split('. ')
        if len(sentences) > 0:
            return f"{sentences[0].strip()}. This covers the essential concepts and implementation details you need to get started."
        else:
            return "Based on the documentation, here are the key concepts and implementation details."
    
    def _structure_content(self, content: str) -> str:
        """Structure content with bullet points and formatting."""
        # Split content into logical chunks
        lines = content.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Add bullet points to important lines
            if len(line) > 20 and not line.startswith('-') and not line.startswith('*'):
                structured_lines.append(f"- {line}")
            else:
                structured_lines.append(line)
        
        return '\n'.join(structured_lines[:5])  # Limit to 5 key points
    
    def _contains_implementation_info(self, contents: List[str]) -> bool:
        """Check if content contains implementation information."""
        combined = ' '.join(contents).lower()
        implementation_keywords = ['install', 'setup', 'configure', 'create', 'run', 'start', 'deploy', 'build']
        return any(keyword in combined for keyword in implementation_keywords)
    
    def _extract_implementation_steps(self, contents: List[str]) -> List[str]:
        """Extract implementation steps from content."""
        combined = ' '.join(contents)
        steps = []
        
        # Look for numbered steps or bullet points
        lines = combined.split('\n')
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                line.startswith(('-', '*')) and len(line) > 10):
                steps.append(line.lstrip('123456789.- *'))
                
        # If no structured steps found, create generic ones
        if not steps:
            steps = [
                "Review the documentation sources below for detailed setup instructions",
                "Follow the quickstart guide if available",
                "Configure the necessary settings and dependencies",
                "Test your implementation with the provided examples"
            ]
        
        return steps[:5]  # Limit to 5 steps
    
    def _extract_code_examples(self, contents: List[str]) -> List[Dict[str, str]]:
        """Extract code examples from content."""
        examples = []
        combined = '\n'.join(contents)
        
        # Simple regex to find code blocks or commands
        import re
        
        # Look for common code patterns
        code_patterns = [
            (r'```(\w+)\n(.*?)```', r'\1', r'\2'),  # Markdown code blocks
            (r'`([^`]+)`', 'bash', r'\1'),  # Inline code
            (r'^npm .*|^pip .*|^docker .*', 'bash', r'\0'),  # Commands
        ]
        
        for pattern, lang_group, code_group in code_patterns:
            matches = re.findall(pattern, combined, re.MULTILINE | re.DOTALL)
            for match in matches[:2]:  # Limit to 2 examples
                if isinstance(match, tuple):
                    lang, code = match[0], match[1]
                else:
                    lang, code = 'bash', match
                    
                if len(code.strip()) > 10:  # Only include substantial code
                    examples.append({
                        'language': lang or 'bash',
                        'code': code.strip()[:300]  # Limit code length
                    })
        
        return examples
    
    def _generate_next_steps(self, contents: List[str], sources: set) -> str:
        """Generate next steps based on content."""
        combined = ' '.join(contents).lower()
        
        steps = []
        
        if 'quickstart' in combined or 'getting started' in combined:
            steps.append("🚀 Follow the quickstart guide mentioned in the sources")
        
        if 'api' in combined:
            steps.append("📚 Review the API documentation for detailed usage")
            
        if 'example' in combined or 'sample' in combined:
            steps.append("💡 Try the examples provided in the documentation")
        
        if 'deploy' in combined:
            steps.append("🔧 Set up your deployment environment")
            
        # Default steps
        if not steps:
            steps.extend([
                "📖 Read through all the source documents linked below",
                "🛠️ Set up your development environment",
                "🧪 Start with a simple proof-of-concept"
            ])
        
        steps.append(f"📋 Consult the {len(sources)} source documents for comprehensive details")
        
        return '\n'.join(f"- {step}" for step in steps)
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Use highest similarity score as base confidence
        max_score = max(result.get('similarity_score', 0.0) for result in search_results)
        
        # Adjust based on number of results and score distribution
        if len(search_results) >= 3:
            # Multiple relevant results increase confidence
            avg_top_3 = sum(result.get('similarity_score', 0.0) for result in search_results[:3]) / 3
            confidence = min(0.95, max_score * 0.7 + avg_top_3 * 0.3)
        else:
            confidence = min(0.85, max_score)
        
        return max(0.0, min(1.0, confidence))
    
    def _prepare_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for response."""
        sources = []
        
        for result in search_results:
            source = {
                'url': result.get('source_url', ''),
                'title': result.get('metadata', {}).get('title', 'Untitled'),
                'snippet': result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', ''),
                'similarity_score': result.get('similarity_score', 0.0),
                'chunk_index': result.get('chunk_index', 0)
            }
            sources.append(source)
        
        return sources
    
    async def batch_query(self, requests: List[QueryRequest]) -> List[QueryResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            requests: List of QueryRequest objects
            
        Returns:
            List of QueryResponse objects
        """
        tasks = [self.query(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch query {i} failed: {response}")
                results.append(QueryResponse(
                    query=requests[i].query,
                    answer=f"Query failed: {str(response)}",
                    sources=[],
                    confidence=0.0
                ))
            else:
                results.append(response)
        
        return results

    async def answer_query_multi_domain(
        self, 
        query: str, 
        domains: List[str], 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer query using multiple domain sources.
        
        Args:
            query: User query
            domains: List of domains to search
            top_k: Number of top results to use
            
        Returns:
            Answer with sources and metadata
        """
        try:
            self.logger.info(f"Starting multi-domain query for domains: {domains}")
            
            # Validate domains first
            self.logger.debug("Validating domains...")
            domain_status = await self.multi_domain_store.validate_domains(domains)
            valid_domains = [d for d, status in domain_status.items() if status]
            
            self.logger.info(f"Domain validation results: {domain_status}")
            self.logger.info(f"Valid domains: {valid_domains}")
            
            if not valid_domains:
                return {
                    "answer": "No domains with embeddings found. Please ensure the selected domains have been crawled and embedded.",
                    "sources": [],
                    "error": "No valid domains found with FAISS indexes",
                    "domain_status": domain_status
                }
            
            # Generate query embedding
            self.logger.debug("Generating query embedding...")
            query_embeddings = await self.embedding_service.generate_embeddings([query])
            if not query_embeddings:
                return {
                    "answer": self._create_fallback_answer([]),
                    "sources": [],
                    "error": "Failed to generate query embedding"
                }
            query_embedding = query_embeddings[0]
            self.logger.debug(f"Generated embedding of dimension: {len(query_embedding)}")
            
            # Search across domains
            self.logger.debug(f"Searching across {len(valid_domains)} domains...")
            search_results = await self.multi_domain_store.search_domains(
                domains=valid_domains,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            self.logger.info(f"Found {len(search_results)} search results across domains")
            
            if not search_results:
                return {
                    "answer": self._create_fallback_answer([]),
                    "sources": [],
                    "note": "No relevant documents found across domains",
                    "domains_searched": valid_domains
                }
            
            # Prepare context for LLM
            context_docs = []
            for result in search_results:
                # Fix URL field mapping - check multiple possible field names
                source_url = result.source_url
                if not source_url or source_url == 'undefined':
                    # Try alternative field names from metadata
                    metadata = result.metadata or {}
                    source_url = (
                        metadata.get('url') or 
                        metadata.get('source_url') or 
                        metadata.get('link') or 
                        metadata.get('page_url') or 
                        metadata.get('original_url') or
                        ''
                    )
                
                context_doc = {
                    "content": result.content,
                    "source_url": source_url,
                    "domain": result.domain,
                    "similarity_score": result.normalized_score,
                    "metadata": result.metadata
                }
                context_docs.append(context_doc)
            
            self.logger.debug(f"Prepared {len(context_docs)} context documents for LLM")
            
            # Generate answer
            self.logger.debug("Generating LLM answer...")
            answer = await self._generate_llm_answer(query, context_docs)
            
            # Prepare sources
            sources = []
            for result in search_results:
                # Fix URL field mapping for sources too
                source_url = result.source_url
                if not source_url or source_url == 'undefined':
                    # Try alternative field names from metadata
                    metadata = result.metadata or {}
                    source_url = (
                        metadata.get('url') or 
                        metadata.get('source_url') or 
                        metadata.get('link') or 
                        metadata.get('page_url') or 
                        metadata.get('original_url') or
                        'No URL available'
                    )
                
                source = {
                    "domain": result.domain,
                    "chunk_id": result.chunk_id,
                    "source_url": source_url,
                    "score": result.normalized_score,
                    "chunk_index": result.chunk_index,
                    "title": result.metadata.get('title', 'Untitled') if result.metadata else 'Untitled',
                    "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content
                }
                sources.append(source)
            
            self.logger.info(f"Multi-domain query completed successfully with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "domains_searched": valid_domains,
                "total_results": len(search_results),
                "query": query,
                "processing_info": {
                    "domains_requested": domains,
                    "domains_validated": domain_status,
                    "domains_searched": valid_domains,
                    "results_per_domain": {
                        domain: len([r for r in search_results if r.domain == domain])
                        for domain in valid_domains
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Multi-domain query failed: {e}", exc_info=True)
            return {
                "answer": self._create_fallback_answer([]),
                "sources": [],
                "error": str(e),
                "query": query,
                "domains_requested": domains
            }
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains for querying."""
        return list(self.vector_stores.keys())
    
    def get_domain_stats(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a domain."""
        if domain not in self.vector_stores:
            return None
        
        return self.vector_stores[domain].get_statistics()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline."""
        return {
            'embedding_model': self.embedding_service.get_model_info(),
            'llm_available': self._llm_client is not None,
            'available_domains': list(self.vector_stores.keys()),
            'total_domains': len(self.vector_stores),
            'domain_stats': {
                domain: store.get_statistics() 
                for domain, store in self.vector_stores.items()
            }

        }