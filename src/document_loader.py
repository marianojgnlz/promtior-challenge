from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Tuple
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from .html_cleaner import HTMLCleaner

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50, 
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""] 
        )
        self.html_cleaner = HTMLCleaner()

    def _is_social_media(self, url: str) -> bool:
        social_domains = [
            'twitter.com', 'x.com',
            'linkedin.com',
            'facebook.com',
            'instagram.com',
            'youtube.com'
        ]
        return any(domain in url.lower() for domain in social_domains)

    def _extract_social_links(self, html_content: str) -> List[str]:
        """Extract social media links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        social_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if self._is_social_media(href):
                social_links.append(href)
        
        return list(set(social_links))  # Remove duplicates

    def _clean_social_media_content(self, url: str, content: str) -> str:
        """Clean and structure social media content based on platform."""
        platform = self._get_platform(url)
        soup = BeautifulSoup(content, 'html.parser')
        cleaned_content = []

        if platform == 'linkedin':
            # Clean LinkedIn content
            # Look for posts, articles, and company updates
            main_content = soup.find('div', {'class': ['feed-shared-update-v2', 'article-content']})
            if main_content:
                # Get post text
                post_text = main_content.find('div', {'class': 'feed-shared-text'})
                if post_text:
                    cleaned_content.append(f"LinkedIn Post: {post_text.get_text().strip()}")
                
                # Get article title and description
                article_title = main_content.find(['h1', 'h2'], {'class': 'article-title'})
                if article_title:
                    cleaned_content.append(f"Article Title: {article_title.get_text().strip()}")

        elif platform == 'twitter':
            tweets = soup.find_all('div', {'class': ['tweet', 'timeline-Tweet']})
            for tweet in tweets:
                tweet_text = tweet.find('div', {'class': ['tweet-text', 'timeline-Tweet-text']})
                if tweet_text:
                    cleaned_content.append(f"Tweet: {tweet_text.get_text().strip()}")

        elif platform == 'facebook':
            posts = soup.find_all('div', {'class': ['userContent', 'post_message']})
            for post in posts:
                cleaned_content.append(f"Facebook Post: {post.get_text().strip()}")

        elif platform == 'youtube':
            title = soup.find('meta', {'property': 'og:title'})
            description = soup.find('meta', {'property': 'og:description'})
            
            if title:
                cleaned_content.append(f"Video Title: {title['content']}")
            if description:
                cleaned_content.append(f"Video Description: {description['content']}")

        if not cleaned_content:
            cleaned_content = [self.html_cleaner.clean_content(content)]

        return "\n\n".join([
            f"Platform: {platform.title()}",
            f"Source: {url}",
            "Content:",
            *cleaned_content
        ])

    def _get_platform(self, url: str) -> str:
        """Determine social media platform from URL."""
        url_lower = url.lower()
        if 'linkedin.com' in url_lower:
            return 'linkedin'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'twitter'
        elif 'facebook.com' in url_lower:
            return 'facebook'
        elif 'instagram.com' in url_lower:
            return 'instagram'
        elif 'youtube.com' in url_lower:
            return 'youtube'
        return 'unknown'

    async def fetch_url(self, url: str, session: aiohttp.ClientSession) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, headers=headers, timeout=15) as response:
                html = await response.text()
                
                # Enhanced social media handling
                if self._is_social_media(url):
                    platform = self._get_platform(url)
                    if platform == 'linkedin':
                        return await self._process_linkedin_content(url, html, session)
                    else:
                        return self._clean_social_media_content(url, html)
                else:
                    return self.html_cleaner.clean_content(html)
        except Exception as e:
            return ""

    async def _process_linkedin_content(self, url: str, html: str, session: aiohttp.ClientSession) -> str:
        """Enhanced LinkedIn content processing with deeper exploration."""
        soup = BeautifulSoup(html, 'html.parser')
        content_parts = []

        content_parts.append(f"Platform: LinkedIn\nSource: {url}\n")

        try:
            company_name = soup.find('h1', {'class': 'org-top-card-summary__title'})
            if company_name:
                content_parts.append(f"Company: {company_name.get_text().strip()}")

            about_section = soup.find('section', {'class': 'org-about-section'})
            if about_section:
                content_parts.append(f"About: {about_section.get_text().strip()}")

            posts = soup.find_all('div', {'class': ['feed-shared-update-v2', 'update-components-text']})
            if posts:
                content_parts.append("\nRecent Updates:")
                for post in posts[:5]:  # Get last 5 posts
                    post_text = post.get_text().strip()
                    if post_text:
                        content_parts.append(f"Post: {post_text}")

            additional_urls = []
            about_url = url + "/about"
            posts_url = url + "/posts"
            additional_urls.extend([about_url, posts_url])

            for add_url in additional_urls:
                try:
                    async with session.get(add_url, timeout=10) as response:
                        add_html = await response.text()
                        add_soup = BeautifulSoup(add_html, 'html.parser')
                        
                        main_content = add_soup.find('main')
                        if main_content:
                            content_parts.append(f"\nContent from {add_url}:")
                            content_parts.append(main_content.get_text().strip())
                except Exception as e:
                    pass

        except Exception as e:
            pass

        return "\n\n".join(filter(None, content_parts))

    async def load_web_documents(self, urls: List[str]) -> List[Document]:
        if not urls:
            return []
            
        documents = []
        social_links = set()
        
        for url in urls:
            try:
                if not self._is_social_media(url):
                    loader = RecursiveUrlLoader(
                        url=url,
                        max_depth=2, 
                        extractor=lambda x: self.html_cleaner.clean_content(x),
                        prevent_outside=True,  
                        exclude_dirs=('login', 'signup', 'cart', 'checkout', 'account'),
                    )
                    
                    site_docs = await asyncio.to_thread(loader.load)
                    documents.extend(site_docs)
                    
                    timeout = aiohttp.ClientTimeout(total=15)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url) as response:
                            html = await response.text()
                            new_social_links = self._extract_social_links(html)
                            social_links.update(new_social_links)
                
                else:
                    timeout = aiohttp.ClientTimeout(total=15)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        content = await self.fetch_url(url, session)
                        if content:
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": url,
                                    "type": "social_media"
                                }
                            ))
            
            except Exception as e:
                pass
        
        if social_links:
            timeout = aiohttp.ClientTimeout(total=15)
            conn = aiohttp.TCPConnector(limit=5)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
                tasks = [self.fetch_url(url, session) for url in social_links]
                contents = await asyncio.gather(*tasks, return_exceptions=True)
                
                for url, content in zip(social_links, contents):
                    if content and isinstance(content, str):
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "source": url,
                                "type": "social_media"
                            }
                        ))
        
        return documents

    def _process_document_language(self, content: str, metadata: Dict) -> Tuple[str, Dict]:
        """Process document language and update metadata accordingly."""
        detected_lang = self._detect_language(content)
        
        metadata['detected_language'] = detected_lang
        
        if 'language' in metadata and metadata['language'] != detected_lang:
            metadata['original_language_meta'] = metadata['language']
            metadata['language'] = detected_lang
        else:
            metadata['language'] = detected_lang

        return content, metadata

    async def process_documents(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
            
        try:
            splits = self.text_splitter.split_documents(docs)
            splits = splits[:50]  
            return splits

        except Exception as e:
            raise e
