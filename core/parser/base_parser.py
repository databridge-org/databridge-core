from abc import ABC, abstractmethod
from typing import List, Union
from fastapi import UploadFile

class BaseParser(ABC):
    """Base class for document parsing"""
    
    @abstractmethod
    async def split_text(self, text: str) -> List[str]:
        """Split plain text into chunks"""
        pass
    
    @abstractmethod
    async def parse_file(self, file: Union[UploadFile, bytes], content_type: str) -> List[str]:
        """Parse file content into text chunks"""
        pass
