from typing import Dict, List, Union
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

@dataclass
class Source:
    url: str
    source_type: str

class SourceProcessor:
    @staticmethod
    def validate_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def process_sources(self, sources: List[Union[str, Path]]) -> Dict[str, List[str]]:
        processed_sources = {
            'urls': [],
            'files': []
        }

        for source in sources:
            if isinstance(source, str) and self.validate_url(source):
                processed_sources['urls'].append(source)
            elif isinstance(source, (str, Path)):
                processed_sources['files'].append(str(source))

        return processed_sources 