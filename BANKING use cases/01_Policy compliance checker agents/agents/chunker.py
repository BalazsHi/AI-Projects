import textwrap
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
class ChunkerAgent:
    """Agent responsible for intelligently chunking documents."""
    def __init__(self, chunk_size: int = 30000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, content: str, doc_type: str = "policy") -> List[Dict]:
        """
        Chunk document into manageable pieces with context preservation.
        """
        try:
            logger.info(f"Chunking document of {len(content)} characters into {self.chunk_size}-char chunks")
        
            if not content or len(content.strip()) < 10:
                logger.error("Empty or too short content provided for chunking")
                return []
        
            # Clean and prepare content
            content = content.strip()
        
            # Split by sections first if possible
            chunks = self._smart_chunk(content)
        
            result = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    result.append({
                        "chunk_id": i + 1,
                        "content": chunk.strip(),
                        "doc_type": doc_type,
                        "char_count": len(chunk),
                        "estimated_tokens": len(chunk) // 4
                    })
        
            logger.info(f"Successfully created {len(result)} chunks")
        
            # Debug: print first chunk preview
            if result:
                logger.info(f"First chunk preview (200 chars): {result[0]['content'][:200]}")
        
            return result
        
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            return []

    def intelligent_chunk(self, content: str, doc_type: str = "policy") -> List[str]:
        """
        Smart chunking that tries to preserve section boundaries.
        Fixed to accept the doc_type parameter that's being passed.
        """
        # Split by common section markers
        section_markers = [
            '\n\n\n',  # Triple newlines
            '\nSection ',
            '\nArticle ',
            '\nChapter ',
            '\n### ',
            '\n## ',
            '\n# '
        ]
    
        # Try to split by sections first
        sections = [content]
    
        for marker in section_markers:
            new_sections = []
            for section in sections:
                if marker in section:
                    parts = section.split(marker)
                    for i, part in enumerate(parts):
                        if i > 0:  # Add marker back except for first part
                            part = marker.lstrip('\n') + part
                        if part.strip():
                            new_sections.append(part)
                else:
                    new_sections.append(section)
            sections = new_sections
    
        # Now chunk sections that are too large
        final_chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                final_chunks.append(section)
            else:
                # Split large sections into smaller chunks
                final_chunks.extend(self._split_large_section(section))
    
        return final_chunks

    def _smart_chunk(self, content: str) -> List[str]:
        """
        Internal method that calls intelligent_chunk for backward compatibility.
        """
        return self.intelligent_chunk(content)

    def _split_large_section(self, section: str) -> List[str]:
        """Split a large section into smaller chunks with overlap."""
        chunks = []
        start = 0
    
        while start < len(section):
            end = start + self.chunk_size
        
            if end >= len(section):
                # Last chunk
                chunks.append(section[start:])
                break
        
            # Try to find a good break point (sentence end, paragraph, etc.)
            chunk = section[start:end]
        
            # Look for sentence endings in the last 200 characters
            break_points = ['. ', '.\n', '!\n', '?\n', '\n\n']
            best_break = end
        
            for point in break_points:
                last_occurrence = chunk.rfind(point)
                if last_occurrence > len(chunk) - 200:  # Within last 200 chars
                    best_break = start + last_occurrence + len(point)
                    break
        
            chunks.append(section[start:best_break])
            start = best_break - self.overlap
        
        return chunks
