# 🔍 Query Documentation Frontend Implementation

> Comprehensive documentation of all implemented logic for the Query Documentation frontend section

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [HTML Structure](#html-structure)
4. [JavaScript Implementation](#javascript-implementation)
5. [CSS Styling](#css-styling)
6. [API Integration](#api-integration)
7. [User Interface Features](#user-interface-features)
8. [Code Examples](#code-examples)

## Overview

The Query Documentation frontend section provides a dual-mode interface for querying documentation:

- **Single Domain Mode**: Query a specific documentation domain
- **Multiple Domain Mode**: Query across multiple selected domains simultaneously

### Key Features
- Mode switching between single and multi-domain queries
- Searchable domain selection with real-time filtering
- Checkbox-based multi-selection with visual tags
- Results display with syntax highlighting
- Progressive loading and error handling
- Responsive design with accessibility features

---

## Architecture

### Component Structure
```
Query Documentation Section
├── Mode Selection Dropdown
├── Single Domain Mode
│   ├── Domain Selector
│   ├── Query Input
│   └── Results Count
├── Multiple Domain Mode
│   ├── Query Input
│   ├── Searchable Domain Dropdown
│   ├── Selected Domains Tags
│   └── Results Count
└── Results Display Area
```

### Data Flow
```
User Input → Domain Selection → Query Submission → API Call → Results Processing → UI Update
```

---

## HTML Structure

### Main Query Section
```html
<section class="operation-section">
    <h2>🔍 Query Documentation</h2>
    
    <!-- Mode Selection -->
    <div class="mode-selection">
        <div class="form-group">
            <label for="queryMode">Query Mode:</label>
            <select id="queryMode">
                <option value="single">Single Domain</option>
                <option value="multiple">Multiple Domains</option>
            </select>
        </div>
    </div>

    <!-- Single Domain Query -->
    <div id="single-domain-mode" class="query-mode-content active">
        <form id="queryForm" class="operation-form">
            <div class="form-row">
                <div class="form-group">
                    <label for="queryDomain">Select Domain:</label>
                    <select id="queryDomain" required>
                        <option value="">Select a domain...</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="topK">Results Count:</label>
                    <input type="number" id="topK" value="5" min="1" max="20">
                </div>
            </div>
            <div class="form-group">
                <label for="queryText">Your Question:</label>
                <textarea id="queryText" required rows="3" 
                         placeholder="How do I implement authentication in the API?"></textarea>
            </div>
            <div class="task-controls">
                <button type="button" id="refreshQueryDomains" class="btn btn-secondary">Refresh</button>
                <button type="submit" class="btn btn-primary">Search Documentation</button>
            </div>
        </form>
    </div>

    <!-- Multi Domain Query -->
    <div id="multi-domain-mode" class="query-mode-content">
        <form id="multiQueryForm" class="operation-form">
            <div class="form-group">
                <label for="multiQueryText">Your Question:</label>
                <textarea id="multiQueryText" required rows="3" 
                         placeholder="How do I implement authentication? (searches across all selected domains)"></textarea>
            </div>
            <div class="form-group">
                <label for="domainSearch">Select Domains:</label>
                <input type="text" id="domainSearch" placeholder="Type to filter domains..." 
                       class="domain-search-input">
                <div id="domainDropdown" class="domain-dropdown">
                    <!-- Domains with checkboxes will appear here -->
                </div>
                <div id="selectedDomains" class="selected-domains-tags">
                    <!-- Selected domains will appear here as tags -->
                </div>
                <div class="selection-info">
                    <span id="selectedCount">0</span> domains selected
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="multiTopK">Results Count:</label>
                    <input type="number" id="multiTopK" value="10" min="1" max="50">
                </div>
                <div class="form-group">
                    <button type="button" id="refreshMultiDomains" class="btn btn-secondary">Refresh Domains</button>
                </div>
            </div>
            <div class="task-controls">
                <button type="submit" class="btn btn-primary" id="multiSearchBtn" disabled>Search Across Domains</button>
            </div>
        </form>
    </div>

    <div id="queryResults" class="results-display"></div>
</section>
```

---

## JavaScript Implementation

### 1. Main Application Logic (`main.js`)

#### Query Form Handler
```javascript
// Handle query form submission
async handleQuerySubmit() {
    const queryData = {
        query: uiManager.elements.queryText.value.trim(),
        domain: uiManager.elements.queryDomain.value.trim(),
        top_k: parseInt(uiManager.elements.topK.value)
    };

    if (!queryData.query) {
        uiManager.showToast('Please enter a query', CONFIG.STATUS_TYPES.ERROR);
        return;
    }

    if (!queryData.domain) {
        uiManager.showToast('Please select a domain', CONFIG.STATUS_TYPES.ERROR);
        return;
    }

    try {
        uiManager.toggleForm('queryForm', true);
        uiManager.showLoading('Searching documentation...');

        const response = await apiClient.queryDocumentation(queryData);
        
        uiManager.showToast('Query completed successfully!', CONFIG.STATUS_TYPES.SUCCESS);
        
        // Display results
        const resultsHTML = uiManager.createQueryResultsHTML(response);
        uiManager.elements.queryResults.innerHTML = resultsHTML;
        uiManager.elements.queryResults.className = 'results-display success';
        
        // Initialize syntax highlighting for code blocks
        uiManager.initializeSyntaxHighlighting();

    } catch (error) {
        console.error('Query failed:', error);
        uiManager.showToast('Query failed: ' + error.message, CONFIG.STATUS_TYPES.ERROR);
        uiManager.updateStatus('queryResults', 
            uiManager.createStatusHTML('Query Failed', error.message, CONFIG.STATUS_TYPES.ERROR));
    } finally {
        uiManager.hideLoading();
        uiManager.toggleForm('queryForm', false);
    }
}
```

#### Multi-Domain Query Handler
```javascript
// Handle multi-domain query submission
async handleMultiDomainQuerySubmit() {
    try {
        const queryText = document.getElementById('multiQueryText').value.trim();
        const topK = parseInt(document.getElementById('multiTopK').value) || 10;

        if (!queryText) {
            uiManager.showToast('Please enter a question', CONFIG.STATUS_TYPES.WARNING);
            return;
        }

        // Get selected domains from the dynamic domain selector
        const selectedDomains = window.dynamicDomainSelector ? 
            window.dynamicDomainSelector.getSelectedDomains() : [];

        if (selectedDomains.length === 0) {
            uiManager.showToast('Please select at least one domain', CONFIG.STATUS_TYPES.WARNING);
            return;
        }

        if (selectedDomains.length > 10) {
            uiManager.showToast('Please select maximum 10 domains', CONFIG.STATUS_TYPES.WARNING);
            return;
        }

        uiManager.showLoading('Searching across selected domains...');

        const queryData = {
            query: queryText,
            domains: selectedDomains,
            top_k: topK
        };

        const result = await apiClient.queryMultiDomain(queryData);
        
        // Display results
        this.displayMultiDomainResults(result);
        
        uiManager.showToast(`Found results across ${result.domains_searched?.length || 0} domains`, CONFIG.STATUS_TYPES.SUCCESS);

    } catch (error) {
        console.error('Multi-domain query failed:', error);
        uiManager.showToast(`Query failed: ${error.message}`, CONFIG.STATUS_TYPES.ERROR);
    } finally {
        uiManager.hideLoading();
    }
}
```

#### Multi-Domain Results Display
```javascript
// Display multi-domain query results
displayMultiDomainResults(result) {
    const resultsContainer = document.getElementById('queryResults');
    if (!resultsContainer) return;

    const html = `
        <div class="results-header">
            <h3>Multi-Domain Search Results</h3>
            <div class="result-stats">
                <span class="stat-item">Query: "${result.query}"</span>
                <span class="stat-item">Domains: ${result.domains_searched?.join(', ') || 'None'}</span>
                <span class="stat-item">Results: ${result.total_results || 0}</span>
            </div>
        </div>
        
        <div class="answer-section multi-domain-result">
            <h4>📄 Answer</h4>
            <div class="answer-content">${marked.parse(result.answer || 'No answer generated.')}</div>
        </div>

        ${result.sources && result.sources.length > 0 ? `
            <div class="sources-section">
                <h4>📚 Sources</h4>
                <div class="sources-list">
                    ${result.sources.map((source, index) => `
                        <div class="source-item multi-domain-result">
                            <div class="source-header">
                                <span class="source-domain">${source.domain}</span>
                                <span class="result-score">Score: ${(source.score * 100).toFixed(1)}%</span>
                                <span class="source-index">#${index + 1}</span>
                            </div>
                            <div class="source-url">
                                <a href="${source.source_url}" target="_blank" rel="noopener noreferrer">
                                    ${source.source_url}
                                </a>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : ''}
    `;

    resultsContainer.innerHTML = html;
    resultsContainer.scrollIntoView({ behavior: 'smooth' });

    // Highlight code blocks
    if (window.hljs) {
        resultsContainer.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }
}
```

### 2. Simple Domain Selector (`dynamic-domains.js`)

#### Core Class Implementation
```javascript
class SimpleDomainSelector {
    constructor() {
        this.allDomains = [];
        this.selectedDomains = [];
        this.init();
    }
    
    init() {
        // Initialize DOM elements
        this.queryModeSelect = document.getElementById('queryMode');
        this.singleModeDiv = document.getElementById('single-domain-mode');
        this.multiModeDiv = document.getElementById('multi-domain-mode');
        this.domainSearchInput = document.getElementById('domainSearch');
        this.domainDropdown = document.getElementById('domainDropdown');
        this.selectedDomainsContainer = document.getElementById('selectedDomains');
        this.selectedCountSpan = document.getElementById('selectedCount');
        this.multiSearchBtn = document.getElementById('multiSearchBtn');
        this.refreshMultiBtn = document.getElementById('refreshMultiDomains');
        
        this.setupEventListeners();
        this.initializeModeDisplay();
        this.loadDomains();
    }
}
```

#### Mode Switching Logic
```javascript
handleModeChange() {
    if (!this.queryModeSelect || !this.singleModeDiv || !this.multiModeDiv) return;
    
    const mode = this.queryModeSelect.value;
    console.log('Mode changed to:', mode);
    
    if (mode === 'single') {
        this.singleModeDiv.style.display = 'block';
        this.multiModeDiv.style.display = 'none';
    } else {
        this.singleModeDiv.style.display = 'none';
        this.multiModeDiv.style.display = 'block';
    }
}
```

#### Domain Loading and Filtering
```javascript
async loadDomains() {
    try {
        console.log('Loading domains...');
        
        if (window.uiManager && uiManager.showLoading) {
            uiManager.showLoading('Loading domains...');
        }
        
        // Use the global API client if available
        let response;
        if (window.apiClient && typeof window.apiClient.makeRequest === 'function') {
            response = await window.apiClient.makeRequest('/domains');
        } else {
            // Fallback to direct fetch
            const apiUrl = document.getElementById('apiUrl')?.value || 'http://localhost:5002';
            const fetchResponse = await fetch(`${apiUrl}/domains`);
            response = { success: fetchResponse.ok, data: await fetchResponse.json() };
        }
        
        if (response.success) {
            this.allDomains = response.data || [];
            console.log('Loaded domains:', this.allDomains);
            this.renderAllDomains();
            
            if (window.uiManager && uiManager.showToast) {
                uiManager.showToast('Domains loaded successfully', 'success');
            }
        }
    } catch (error) {
        console.error('Error loading domains:', error);
        this.allDomains = [];
        this.renderAllDomains();
    } finally {
        if (window.uiManager && uiManager.hideLoading) {
            uiManager.hideLoading();
        }
    }
}

filterDomains(searchTerm) {
    if (!this.domainDropdown) return;
    
    this.domainDropdown.innerHTML = '';
    
    const filteredDomains = this.allDomains.filter(domain => 
        domain.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    if (filteredDomains.length === 0) {
        const emptyItem = document.createElement('div');
        emptyItem.className = 'domain-dropdown-item disabled';
        emptyItem.innerHTML = '<span class="domain-name">No domains found</span>';
        this.domainDropdown.appendChild(emptyItem);
    } else {
        filteredDomains.forEach(domain => {
            this.createDomainItem(domain, searchTerm);
        });
    }
}
```

#### Domain Selection Management
```javascript
createDomainItem(domain, searchTerm = '') {
    const item = document.createElement('div');
    item.className = 'domain-dropdown-item';
    
    // Highlight search term
    let displayName = domain;
    if (searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        displayName = domain.replace(regex, '<mark>$1</mark>');
    }
    
    const isSelected = this.selectedDomains.includes(domain);
    
    item.innerHTML = `
        <input type="checkbox" class="domain-checkbox" value="${domain}" id="domain-${domain}" ${isSelected ? 'checked' : ''}>
        <label for="domain-${domain}" class="domain-name">${displayName}</label>
    `;
    
    const checkbox = item.querySelector('.domain-checkbox');
    checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            this.selectDomain(domain);
        } else {
            this.removeDomain(domain);
        }
    });
    
    // Make entire item clickable
    item.addEventListener('click', (e) => {
        if (e.target.type !== 'checkbox' && e.target.tagName !== 'LABEL') {
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
    
    this.domainDropdown.appendChild(item);
}

selectDomain(domain) {
    if (!this.selectedDomains.includes(domain) && this.selectedDomains.length < 10) {
        this.selectedDomains.push(domain);
        this.updateSelectedDomainsDisplay();
        this.updateSearchButton();
    } else if (this.selectedDomains.length >= 10) {
        if (window.uiManager && uiManager.showToast) {
            uiManager.showToast('Maximum 10 domains can be selected', 'warning');
        }
    }
}

removeDomain(domain) {
    this.selectedDomains = this.selectedDomains.filter(d => d !== domain);
    this.updateSelectedDomainsDisplay();
    this.updateSearchButton();
}
```

### 3. API Client Integration (`api.js`)

#### Single Domain Query
```javascript
// Query documentation
async queryDocumentation(queryData) {
    return await this.request(CONFIG.ENDPOINTS.QUERY, {
        method: 'POST',
        body: JSON.stringify(queryData)
    });
}
```

#### Multi-Domain Query
```javascript
// Query multiple domains simultaneously
async queryMultiDomain(queryData) {
    return await this.request('/query-multi-domain', {
        method: 'POST',
        body: JSON.stringify({
            query: queryData.query,
            domains: queryData.domains, // Array of domain names
            top_k: queryData.top_k || 5
        })
    });
}
```

#### Domain Management
```javascript
// Get available domains
async getDomains() {
    return await this.request(CONFIG.ENDPOINTS.DOMAINS);
}

// Get available domains for selection
async getAvailableDomains() {
    try {
        const response = await this.request('/domains/available');
        return response.domains || [];
    } catch (error) {
        console.error('Failed to fetch available domains:', error);
        throw error;
    }
}

// Validate domains before querying
async validateDomains(domains) {
    return await this.request('/domains/validate', {
        method: 'POST',
        body: JSON.stringify({ domains })
    });
}
```

### 4. UI Management (`ui.js`)

#### Results HTML Generation
```javascript
// Create query results HTML
createQueryResultsHTML(results) {
    if (!results.answer && (!results.sources || results.sources.length === 0)) {
        return this.createStatusHTML('No Results', 'No results found for your query.', CONFIG.STATUS_TYPES.WARNING);
    }

    let html = '<div class="query-results">';
    
    // Main answer
    if (results.answer) {
        html += '<div class="answer-section">';
        html += '<h3>📝 Answer</h3>';
        html += `<div class="answer-content">${this.formatAnswerContent(results.answer)}</div>`;
        html += '</div>';
    }

    // Sources
    if (results.sources && results.sources.length > 0) {
        html += '<div class="sources-section">';
        html += '<h3>📚 Sources</h3>';
        html += '<div class="sources-list">';
        
        results.sources.forEach((source, index) => {
            html += '<div class="source-item">';
            const score = source.similarity_score || source.score || 0;
            html += `<h4>Source ${index + 1} (Score: ${score.toFixed(3)})</h4>`;
            html += `<p class="source-url"><strong>URL:</strong> <a href="${source.url}" target="_blank">${source.url}</a></p>`;
            if (source.title) {
                html += `<p class="source-title"><strong>Title:</strong> ${source.title}</p>`;
            }
            const content = source.snippet || source.content || 'No content available';
            html += `<div class="source-content">${content}</div>`;
            html += '</div>';
        });
        
        html += '</div>';
        html += '</div>';
    }

    html += '</div>';
    return html;
}
```

---

## CSS Styling

### Mode Selection Styles
```css
/* Mode Selection Styles */
.mode-selection {
    margin-bottom: var(--spacing-lg);
    padding: var(--spacing-md);
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
}

.mode-selection .form-group {
    margin-bottom: 0;
}

.mode-selection select {
    min-width: 200px;
    font-weight: 500;
}

/* Query Mode Content */
.query-mode-content {
    display: none;
}

.query-mode-content.active {
    display: block;
}
```

### Domain Dropdown Styles
```css
/* Dynamic Multi-Domain Selector Styles */
.dynamic-domain-selector {
    position: relative;
    margin-bottom: var(--spacing-md);
}

.domain-dropdown {
    background: var(--bg-primary);
    border: 2px solid var(--border-color);
    border-radius: var(--radius-md);
    max-height: 200px;
    overflow-y: auto;
    z-index: 1000;
    box-shadow: var(--shadow-md);
    margin-top: var(--spacing-sm);
}

.domain-dropdown-item {
    display: flex;
    align-items: center;
    padding: var(--spacing-sm) var(--spacing-md);
    cursor: pointer;
    transition: var(--transition-fast);
    border-bottom: 1px solid var(--bg-secondary);
}

.domain-dropdown-item:hover {
    background: var(--bg-secondary);
}

.domain-dropdown-item.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: var(--bg-tertiary);
}
```

### Selected Domain Tags
```css
/* Selected Domains Tags */
.selected-domains-tags {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
    margin: var(--spacing-md) 0;
    min-height: 40px;
    padding: var(--spacing-sm);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
}

.domain-tag {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-xs) var(--spacing-md);
    background: var(--primary-color);
    color: white;
    border-radius: var(--radius-md);
    font-size: 0.875rem;
    font-weight: 500;
    animation: fadeIn 0.2s ease-in-out;
}

.domain-tag-remove {
    background: none;
    border: none;
    color: white;
    cursor: pointer;
    padding: 0;
    width: 16px;
    height: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: var(--transition-fast);
    font-size: 12px;
    font-weight: bold;
}

.domain-tag-remove:hover {
    background: rgba(255, 255, 255, 0.2);
}
```

### Multi-Domain Results
```css
/* Multi-Domain Results Styles */
.multi-domain-result {
    border-left: 4px solid var(--primary-color);
    padding-left: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
}

.source-domain {
    display: inline-block;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: var(--spacing-sm);
}

.result-score {
    display: inline-block;
    background: rgba(37, 99, 235, 0.1);
    color: var(--primary-color);
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: var(--spacing-sm);
}
```

---

## API Integration

### Endpoint Mappings
- **Single Domain Query**: `POST /query`
- **Multi-Domain Query**: `POST /query-multi-domain`
- **Get Domains**: `GET /domains`
- **Validate Domains**: `POST /domains/validate`

### Request/Response Formats

#### Single Domain Query Request
```json
{
    "query": "How do I implement authentication?",
    "domain": "docs-example-com",
    "top_k": 5
}
```

#### Multi-Domain Query Request
```json
{
    "query": "How do I implement authentication?",
    "domains": ["docs-example-com", "api-docs-example"],
    "top_k": 10
}
```

#### Query Response Format
```json
{
    "query": "How do I implement authentication?",
    "answer": "To implement authentication...",
    "sources": [
        {
            "domain": "docs-example-com",
            "score": 0.95,
            "source_url": "https://docs.example.com/auth",
            "title": "Authentication Guide",
            "content": "Authentication content..."
        }
    ],
    "domains_searched": ["docs-example-com"],
    "total_results": 5
}
```

---

## User Interface Features

### 1. Mode Switching
- **Single Domain**: Traditional query interface with domain dropdown
- **Multiple Domain**: Advanced interface with searchable domain selection

### 2. Domain Selection
- **Real-time Search**: Filter domains as you type
- **Checkbox Selection**: Click anywhere on item to toggle selection
- **Visual Tags**: Selected domains appear as removable chips
- **Maximum Limit**: Enforces 10-domain maximum with user feedback

### 3. Results Display
- **Syntax Highlighting**: Code blocks highlighted with highlight.js
- **Markdown Rendering**: Results processed with marked.js
- **Responsive Layout**: Mobile-friendly design
- **Progressive Loading**: Loading states and error handling

### 4. User Experience
- **Auto-complete**: Domain name suggestions
- **Validation**: Real-time form validation
- **Accessibility**: Keyboard navigation and screen reader support
- **Performance**: Debounced search and efficient rendering

---

## Code Examples

### Complete Form Submission Flow

```javascript
// 1. User submits multi-domain query
document.getElementById('multiQueryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // 2. Validate input
    const queryText = document.getElementById('multiQueryText').value.trim();
    if (!queryText) {
        uiManager.showToast('Please enter a question', 'warning');
        return;
    }
    
    // 3. Get selected domains
    const selectedDomains = window.dynamicDomainSelector.getSelectedDomains();
    if (selectedDomains.length === 0) {
        uiManager.showToast('Please select at least one domain', 'warning');
        return;
    }
    
    // 4. Show loading state
    uiManager.showLoading('Searching across selected domains...');
    
    try {
        // 5. Submit query
        const result = await apiClient.queryMultiDomain({
            query: queryText,
            domains: selectedDomains,
            top_k: parseInt(document.getElementById('multiTopK').value) || 10
        });
        
        // 6. Display results
        app.displayMultiDomainResults(result);
        uiManager.showToast(`Found results across ${result.domains_searched?.length || 0} domains`, 'success');
        
    } catch (error) {
        uiManager.showToast(`Query failed: ${error.message}`, 'error');
    } finally {
        uiManager.hideLoading();
    }
});
```

### Dynamic Domain Item Creation

```javascript
// Create interactive domain selection item
createDomainItem(domain, searchTerm = '') {
    const item = document.createElement('div');
    item.className = 'domain-dropdown-item';
    
    // Highlight matching text
    let displayName = domain;
    if (searchTerm) {
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        displayName = domain.replace(regex, '<mark>$1</mark>');
    }
    
    const isSelected = this.selectedDomains.includes(domain);
    
    item.innerHTML = `
        <input type="checkbox" class="domain-checkbox" value="${domain}" 
               id="domain-${domain}" ${isSelected ? 'checked' : ''}>
        <label for="domain-${domain}" class="domain-name">${displayName}</label>
    `;
    
    // Add event listeners
    const checkbox = item.querySelector('.domain-checkbox');
    checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
            this.selectDomain(domain);
        } else {
            this.removeDomain(domain);
        }
    });
    
    // Make entire item clickable
    item.addEventListener('click', (e) => {
        if (e.target.type !== 'checkbox' && e.target.tagName !== 'LABEL') {
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        }
    });
    
    return item;
}
```

---

## Conclusion

This implementation provides a comprehensive, user-friendly interface for querying documentation across single or multiple domains. The modular architecture ensures maintainability and extensibility, while the responsive design and accessibility features provide an excellent user experience across all devices and user types.

### Key Achievements
- ✅ Dual-mode query interface
- ✅ Real-time domain search and filtering
- ✅ Visual domain selection with tags
- ✅ Comprehensive error handling
- ✅ Responsive and accessible design
- ✅ Integration with existing API architecture
- ✅ Progressive loading and user feedback
- ✅ Syntax highlighting and markdown rendering

### Files Involved
- `dfrontend/index.html` - Main structure
- `dfrontend/js/main.js` - Core application logic
- `dfrontend/js/dynamic-domains.js` - Domain selection component
- `dfrontend/js/api.js` - API communication
- `dfrontend/js/ui.js` - UI management
- `dfrontend/css/styles.css` - Complete styling

---

*Last Updated: October 6, 2025*
*Documentation Crawler Frontend - Query Implementation Guide*