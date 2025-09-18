import json
import logging
from typing import List, Dict, Any, Optional
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

logger = logging.getLogger(__name__)

class RequirementExtractorAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=4000
        )
        
        # Initialize text splitter for handling long chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ".", " ", ""]
        )
        
        # Enhanced preprocessing patterns
        self.noise_patterns = [
            r'\b(page \d+|pg\. \d+)\b',  # Page numbers
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'^\s*[\d\.\-\(\)]+\s*$',  # Standalone numbers/bullets
            r'\b(figure|table|chart|appendix)\s+\d+\b',  # References to figures/tables
            r'\s{3,}',  # Multiple whitespaces
            r'_{3,}',  # Multiple underscores
        ]
        
        # Regulatory keywords for better context understanding
        self.regulatory_keywords = {
            'mandatory': ['must', 'shall', 'required', 'mandatory', 'obligated', 'necessary'],
            'prohibitive': ['prohibited', 'forbidden', 'not permitted', 'shall not', 'must not'],
            'conditional': ['should', 'may', 'could', 'recommended', 'advisable'],
            'temporal': ['immediately', 'within', 'by', 'before', 'after', 'during'],
            'quantitative': ['minimum', 'maximum', 'at least', 'no more than', 'exceeds', 'below']
        }
    
    def _preprocess_content(self, content: str) -> str:
        """Enhanced content preprocessing to improve parsing accuracy."""
        if not content.strip():
            return content
            
        # Remove common noise patterns
        for pattern in self.noise_patterns:
            content = re.sub(pattern, ' ', content, flags=re.IGNORECASE)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Fix common OCR/parsing issues
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Separate camelCase
        content = re.sub(r'(\d)([A-Za-z])', r'\1 \2', content)  # Separate digits from letters
        content = re.sub(r'([A-Za-z])(\d)', r'\1 \2', content)  # Separate letters from digits
        
        # Preserve sentence structure
        content = re.sub(r'\.(?=[A-Z])', '. ', content)
        
        return content.strip()
    
    def _assess_content_quality(self, content: str) -> Dict[str, Any]:
        """Assess content quality to determine extraction strategy."""
        words = content.split()
        
        # Calculate various quality metrics
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        
        # Count regulatory indicators
        regulatory_count = 0
        for category, keywords in self.regulatory_keywords.items():
            for keyword in keywords:
                regulatory_count += len(re.findall(rf'\b{keyword}\b', content, re.IGNORECASE))
        
        # Calculate sentence coherence (basic metric)
        sentences = re.split(r'[.!?]+', content)
        coherent_sentences = sum(1 for s in sentences if len(s.split()) > 3)
        
        # Detect potential parsing issues
        special_char_ratio = len(re.findall(r'[^\w\s.,!?;:-]', content)) / max(len(content), 1)
        
        quality_score = min(100, (
            (regulatory_count * 10) +
            (coherent_sentences * 5) +
            (50 if 50 < word_count < 1000 else 20) +
            (20 if avg_word_length > 3 else 10) -
            (special_char_ratio * 100)
        ))
        
        return {
            'word_count': word_count,
            'regulatory_indicators': regulatory_count,
            'coherent_sentences': coherent_sentences,
            'quality_score': quality_score,
            'needs_splitting': word_count > 2500,
            'likely_corrupted': special_char_ratio > 0.3 or quality_score < 20
        }
    
    def extract_requirements(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract regulatory requirements from a document chunk with enhanced processing."""
        try:
            content = chunk.get('content', '')
            chunk_id = chunk.get('chunk_id', 'unknown')
            
            if not content.strip():
                logger.warning(f"Empty content in chunk {chunk_id}")
                return []
            
            # Enhanced preprocessing
            processed_content = self._preprocess_content(content)
            
            # Assess content quality
            quality_assessment = self._assess_content_quality(processed_content)
            logger.info(f"Content quality for chunk {chunk_id}: {quality_assessment}")
            
            # Handle corrupted or poor quality content
            if quality_assessment['likely_corrupted']:
                logger.warning(f"Detected corrupted content in chunk {chunk_id}, using aggressive fallback")
                return self._extract_fallback_requirements(processed_content, chunk_id, aggressive=True)
            
            # Split large chunks for better processing
            if quality_assessment['needs_splitting']:
                return self._process_large_chunk(processed_content, chunk_id)
            
            # Standard extraction with enhanced prompting
            prompt = self._create_enhanced_extraction_prompt(processed_content, chunk_id, quality_assessment)
            
            # Multiple extraction attempts with different strategies
            for attempt in range(2):
                try:
                    response = self.llm.invoke([
                        SystemMessage(content=self._get_enhanced_system_prompt()),
                        HumanMessage(content=prompt)
                    ])
                    
                    response_text = response.content.strip()
                    logger.info(f"Raw response for chunk {chunk_id} (attempt {attempt + 1}): {response_text[:200]}...")
                    
                    # Enhanced JSON parsing
                    requirements = self._parse_enhanced_json_response(response_text, chunk_id)
                    
                    if requirements or attempt == 1:  # Use results even if empty on final attempt
                        logger.info(f"Extracted {len(requirements)} requirements from chunk {chunk_id}")
                        return requirements
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for chunk {chunk_id}: {str(e)}")
                    if attempt == 1:  # Final attempt failed
                        raise e
            
            return []
            
        except Exception as e:
            logger.error(f"Error extracting requirements from chunk {chunk_id}: {str(e)}")
            return self._create_error_fallback(chunk, str(e))
    
    def _process_large_chunk(self, content: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Process large chunks by splitting them into smaller pieces."""
        logger.info(f"Processing large chunk {chunk_id} with splitting")
        
        sub_chunks = self.text_splitter.split_text(content)
        all_requirements = []
        
        for i, sub_chunk in enumerate(sub_chunks):
            sub_chunk_id = f"{chunk_id}_SUB{i+1}"
            
            # Create temporary chunk object
            temp_chunk = {
                'content': sub_chunk,
                'chunk_id': sub_chunk_id
            }
            
            # Recursively process sub-chunk
            sub_requirements = self.extract_requirements(temp_chunk)
            
            # Update parent chunk ID
            for req in sub_requirements:
                req['parent_chunk_id'] = chunk_id
                req['sub_chunk_index'] = i + 1
            
            all_requirements.extend(sub_requirements)
        
        return all_requirements
    
    def _get_enhanced_system_prompt(self) -> str:
        """Get enhanced system prompt with better instructions."""
        return """You are an expert regulatory compliance analyst specializing in financial regulations. 

Your task is to extract specific, actionable compliance requirements from regulatory documents. 

EXPERTISE AREAS:
- Banking regulations (Basel III, CRD IV, etc.)
- Risk management frameworks
- Capital adequacy requirements
- Reporting and disclosure obligations
- Governance and operational requirements

EXTRACTION PRINCIPLES:
1. Focus ONLY on explicit obligations (must, shall, required, etc.)
2. Distinguish between mandatory requirements and recommendations
3. Preserve exact regulatory language when possible
4. Identify specific compliance actions, not general descriptions
5. Extract quantitative thresholds, deadlines, and specific criteria

Always respond with valid JSON only. Be precise and comprehensive."""
    
    def _create_enhanced_extraction_prompt(self, content: str, chunk_id: str, quality_assessment: Dict) -> str:
        """Create enhanced extraction prompt with adaptive instructions."""
        
        # Adjust instructions based on content quality
        if quality_assessment['regulatory_indicators'] > 10:
            extraction_focus = "This appears to be regulatory-dense content. Pay special attention to compliance obligations."
        elif quality_assessment['word_count'] < 100:
            extraction_focus = "This is a short text segment. Extract any explicit requirements, even brief ones."
        else:
            extraction_focus = "Analyze this content thoroughly for any compliance requirements."
        
        return f"""
{extraction_focus}

REGULATORY TEXT:
{content}

EXTRACTION TASK:
Extract all specific compliance requirements that financial institutions must follow.

MANDATORY INDICATORS TO LOOK FOR:
- "must", "shall", "required to", "obligated to", "mandatory"
- "prohibited from", "shall not", "must not", "forbidden"
- Specific deadlines, thresholds, or quantitative requirements
- Reporting obligations and submission requirements
- Risk management and control requirements

JSON RESPONSE FORMAT:
{{
    "chunk_id": "{chunk_id}",
    "requirements": [
        {{
            "id": "REQ001",
            "requirement": "Complete exact requirement text with all details",
            "category": "risk_management|capital_adequacy|reporting|governance|operational|liquidity|credit_risk|market_risk|compliance",
            "priority": "high|medium|low",
            "reference": "Specific section/paragraph/article reference",
            "keywords": ["relevant", "keywords", "extracted"],
            "requirement_type": "mandatory|prohibitive|conditional|quantitative|procedural",
            "deadline": "If any deadline mentioned",
            "applies_to": "Who this requirement applies to"
        }}
    ]
}}

CRITICAL INSTRUCTIONS:
- Respond with ONLY the JSON object
- No markdown, no explanations, no additional text
- Include ALL explicit requirements found
- If no requirements found, return empty requirements array
- Ensure all JSON is properly escaped and formatted
"""
    
    def _parse_enhanced_json_response(self, response_text: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Enhanced JSON parsing with multiple fallback strategies."""
        try:
            # Clean the response more aggressively
            cleaned_response = self._clean_json_response(response_text)
            
            # Try standard JSON parsing
            parsed_data = json.loads(cleaned_response)
            
            # Validate and extract requirements
            if isinstance(parsed_data, dict) and 'requirements' in parsed_data:
                requirements = parsed_data['requirements']
                if isinstance(requirements, list):
                    return self._validate_enhanced_requirements(requirements, chunk_id)
            
            # Try alternative parsing strategies
            return self._alternative_json_parsing(response_text, chunk_id)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in chunk {chunk_id}: {str(e)}")
            return self._alternative_json_parsing(response_text, chunk_id)
    
    def _clean_json_response(self, response_text: str) -> str:
        """Aggressively clean JSON response."""
        # Remove markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response_text.strip())
        
        # Remove any text before first {
        first_brace = cleaned.find('{')
        if first_brace > 0:
            cleaned = cleaned[first_brace:]
        
        # Remove any text after last }
        last_brace = cleaned.rfind('}')
        if last_brace > 0:
            cleaned = cleaned[:last_brace + 1]
        
        # Fix common JSON issues
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)  # Remove trailing commas
        cleaned = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1 \2', cleaned)  # Fix broken strings
        
        return cleaned
    
    def _alternative_json_parsing(self, response_text: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Alternative JSON parsing strategies."""
        try:
            # Try to find JSON-like structures using regex
            json_matches = re.findall(r'\{[^{}]*"requirement"[^{}]*\}', response_text, re.DOTALL)
            
            requirements = []
            for i, match in enumerate(json_matches):
                try:
                    req_data = json.loads(match)
                    if isinstance(req_data, dict) and 'requirement' in req_data:
                        req_data['id'] = req_data.get('id', f'{chunk_id}_ALT{i+1:03d}')
                        requirements.append(req_data)
                except:
                    continue
            
            if requirements:
                return self._validate_enhanced_requirements(requirements, chunk_id)
                
        except Exception as e:
            logger.warning(f"Alternative JSON parsing failed for chunk {chunk_id}: {str(e)}")
        
        # Final fallback to text-based extraction
        return self._extract_fallback_requirements(response_text, chunk_id)
    
    def _validate_enhanced_requirements(self, requirements: List[Dict], chunk_id: str) -> List[Dict[str, Any]]:
        """Enhanced requirement validation with more fields."""
        validated = []
        
        for i, req in enumerate(requirements):
            if not isinstance(req, dict):
                continue
                
            # Ensure required fields
            requirement_text = req.get('requirement', '').strip()
            if not requirement_text or len(requirement_text) < 10:
                continue
            
            # Enhanced validation
            validated_req = {
                'id': req.get('id', f'{chunk_id}_REQ{i+1:03d}'),
                'requirement': requirement_text,
                'category': req.get('category', 'general'),
                'priority': req.get('priority', 'medium'),
                'reference': req.get('reference', f'Chunk {chunk_id}'),
                'keywords': req.get('keywords', []),
                'requirement_type': req.get('requirement_type', 'mandatory'),
                'deadline': req.get('deadline', ''),
                'applies_to': req.get('applies_to', ''),
                'chunk_id': chunk_id,
                'extraction_method': 'enhanced_llm'
            }
            
            # Add content hash for deduplication
            content_hash = hashlib.md5(requirement_text.encode()).hexdigest()[:8]
            validated_req['content_hash'] = content_hash
            
            validated.append(validated_req)
        
        return validated
    
    def _extract_fallback_requirements(self, response_text: str, chunk_id: str, aggressive: bool = False) -> List[Dict[str, Any]]:
        """Enhanced fallback extraction with better pattern matching."""
        logger.info(f"Using {'aggressive' if aggressive else 'standard'} fallback extraction for chunk {chunk_id}")
        
        requirements = []
        
        # Enhanced requirement patterns
        if aggressive:
            requirement_patterns = [
                r'([^.]*(?:must|shall|required to|obligated to|mandatory)[^.]*\.)',
                r'([^.]*(?:prohibited|forbidden|not permitted|shall not|must not)[^.]*\.)',
                r'([^.]*(?:minimum|maximum|at least|no more than)[^.]*\.)',
                r'([^.]*(?:within \d+|by [A-Za-z]+ \d+|before [A-Za-z]+)[^.]*\.)',
            ]
        else:
            requirement_patterns = [
                r'((?:Banks|Institutions|Entities)[^.]*(?:must|shall|required)[^.]*\.)',
                r'((?:The|A|An)[^.]*(?:requirement|obligation)[^.]*\.)',
                r'([^.]*(?:compliance with|adherence to)[^.]*\.)',
            ]
        
        req_id = 1
        for pattern in requirement_patterns:
            matches = re.finditer(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            for match in matches:
                requirement_text = match.group(1).strip()
                if len(requirement_text) > 20 and len(requirement_text) < 500:
                    
                    # Determine priority based on indicators
                    priority = 'high' if any(word in requirement_text.lower() 
                                           for word in ['must', 'shall', 'mandatory', 'required']) else 'medium'
                    
                    # Determine category based on keywords
                    category = self._classify_requirement_category(requirement_text)
                    
                    requirements.append({
                        'id': f'{chunk_id}_FALLBACK{req_id:03d}',
                        'requirement': requirement_text,
                        'category': category,
                        'priority': priority,
                        'reference': f'Chunk {chunk_id}',
                        'keywords': self._extract_enhanced_keywords(requirement_text),
                        'chunk_id': chunk_id,
                        'extraction_method': 'enhanced_fallback',
                        'content_hash': hashlib.md5(requirement_text.encode()).hexdigest()[:8]
                    })
                    req_id += 1
        
        if not requirements and not aggressive:
            # Try aggressive fallback
            return self._extract_fallback_requirements(response_text, chunk_id, aggressive=True)
        
        if not requirements:
            # Last resort: create a requirement from the chunk content
            content_summary = response_text[:300] + "..." if len(response_text) > 300 else response_text
            requirements.append({
                'id': f'{chunk_id}_SUMMARY001',
                'requirement': f'Manual review required - potential regulatory content: {content_summary}',
                'category': 'review_required',
                'priority': 'low',
                'reference': f'Chunk {chunk_id}',
                'keywords': ['manual_review'],
                'chunk_id': chunk_id,
                'extraction_method': 'summary_fallback'
            })
        
        logger.info(f"Fallback extracted {len(requirements)} requirements from chunk {chunk_id}")
        return requirements
    
    def _classify_requirement_category(self, text: str) -> str:
        """Classify requirement category based on content analysis."""
        text_lower = text.lower()
        
        category_keywords = {
            'capital_adequacy': ['capital', 'tier 1', 'tier 2', 'capital ratio', 'adequacy', 'buffer'],
            'risk_management': ['risk', 'risk management', 'risk assessment', 'risk control', 'exposure'],
            'reporting': ['report', 'reporting', 'disclosure', 'submit', 'filing', 'notification'],
            'liquidity': ['liquidity', 'liquid assets', 'funding', 'cash', 'liquidity ratio'],
            'governance': ['governance', 'board', 'management', 'oversight', 'supervision'],
            'operational': ['operational', 'procedures', 'controls', 'processes', 'systems'],
            'credit_risk': ['credit risk', 'lending', 'loan', 'default', 'credit assessment'],
            'market_risk': ['market risk', 'trading', 'market exposure', 'position'],
            'compliance': ['compliance', 'regulatory', 'regulation', 'authorized', 'license']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_enhanced_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction with better relevance."""
        keywords = set()
        text_lower = text.lower()
        
        # Regulatory action words
        action_keywords = ['assess', 'monitor', 'report', 'maintain', 'establish', 'ensure', 'implement', 'comply']
        for keyword in action_keywords:
            if keyword in text_lower:
                keywords.add(keyword)
        
        # Financial terms
        financial_terms = ['capital', 'risk', 'liquidity', 'exposure', 'ratio', 'threshold', 'limit', 'buffer']
        for term in financial_terms:
            if term in text_lower:
                keywords.add(term)
        
        # Regulatory entities
        entities = ['bank', 'institution', 'entity', 'firm', 'organization']
        for entity in entities:
            if entity in text_lower:
                keywords.add(entity)
        
        # Time-related keywords
        time_patterns = re.findall(r'\b(annual|quarterly|monthly|daily|immediate|within \d+)\b', text_lower)
        keywords.update(time_patterns)
        
        return list(keywords)[:10]  # Limit to 10 most relevant keywords
    
    def _create_error_fallback(self, chunk: Dict[str, Any], error_msg: str) -> List[Dict[str, Any]]:
        """Create enhanced error fallback with better error handling."""
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        return [{
            'id': f'{chunk_id}_ERROR001',
            'requirement': f'EXTRACTION ERROR - Manual review required: {error_msg}. Content preview: {chunk.get("content", "")[:200]}...',
            'category': 'error',
            'priority': 'high',
            'reference': f'Chunk {chunk_id}',
            'keywords': ['manual_review', 'extraction_error', 'system_failure'],
            'chunk_id': chunk_id,
            'extraction_method': 'error_fallback',
            'error': error_msg,
            'requires_manual_review': True
        }]
