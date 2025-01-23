from bs4 import BeautifulSoup
import re

class HTMLCleaner:
    @staticmethod
    def clean_content(html_content: str) -> str:
        """Extract clean, relevant text content from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for link in soup.find_all('a'):
            if link.string:
                link.string.replace_with(f" {link.get_text()} ")
        
        for element in soup.find_all(['script', 'style', 'iframe']):
            element.decompose()
        
        doc_content = HTMLCleaner._find_main_content(soup)
        formatted_content = HTMLCleaner._process_content(doc_content)
        
        seen_content = set()
        cleaned_lines = []
        
        for line in formatted_content.split('\n'):
            line = line.strip()
            if line and line not in seen_content:
                seen_content.add(line)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _find_main_content(soup: BeautifulSoup) -> BeautifulSoup:
        """Find the main content section of the HTML."""
        return (
            soup.find('div', {'id': 'doc-content'}) or
            soup.find('div', {'id': 'main-content'}) or
            soup.find('article') or
            soup.find('main') or
            soup.find('div', class_='content') or
            soup
        )
    
    @staticmethod
    def _process_content(content: BeautifulSoup) -> str:
        """Process and format the content."""
        formatted_content = []
        current_section = None
        section_content = []
        
        def get_clean_text(elem):
            # Handle links
            for link in elem.find_all('a'):
                if link.string:
                    link.string.replace_with(f" {link.get_text()} ")
            
            # Get text and clean it
            text = elem.get_text()
            # Remove extra whitespace and normalize
            text = ' '.join(text.split()).strip()
            # Remove any remaining list markers or special characters
            text = re.sub(r'^[•·⋅∙◦⦁◆►▸▹▻▷▶]', '', text).strip()
            return text
        
        def process_element(element):
            nonlocal current_section, section_content
            
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_section and section_content:
                    formatted_content.extend(['', f'{current_section}', ''])
                    formatted_content.extend(section_content)
                    section_content = []
                current_section = get_clean_text(element)
                
            elif element.name in ['p', 'span']:
                text = get_clean_text(element)
                if text:
                    section_content.append(text)
                    
            elif element.name in ['ul', 'ol']:
                list_items = []
                for li in element.find_all('li', recursive=False):
                    text = get_clean_text(li)
                    if text:
                        list_items.append(text)  # No bullet point, just the text
                if list_items:
                    section_content.extend(list_items)
                    
            elif element.name in ['pre', 'code']:
                text = get_clean_text(element)
                if text:
                    section_content.append(f"CODE: {text}")
        
        # Process all elements
        for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'span', 'pre', 'code'], recursive=True):
            process_element(element)
        
        # Add the last section
        if current_section and section_content:
            formatted_content.extend(['', f'{current_section}', ''])
            formatted_content.extend(section_content)
        
        if not formatted_content and section_content:
            formatted_content.extend(section_content)
        
        return '\n'.join(formatted_content) 