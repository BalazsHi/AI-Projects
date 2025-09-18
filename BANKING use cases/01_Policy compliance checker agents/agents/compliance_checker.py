from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ComplianceCheckerAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.1
        )
    
    def check_compliance(self, bank_policy: str, requirements) -> Dict[str, Any]:
        """Check bank policy compliance against requirements."""
    
        # Handle different input formats
        if isinstance(requirements, dict):
            # Original expected format: {"requirements": [...], "section_info": "..."}
            req_list = requirements.get('requirements', [])
            section_info = requirements.get('section_info', 'Unknown')
            logger.info(f"Checking compliance for {len(req_list)} requirements (dict format)")
        elif isinstance(requirements, list):
            # New format: direct list of requirements
            req_list = requirements
            section_info = 'Unknown'
            logger.info(f"Checking compliance for {len(req_list)} requirements (list format)")
        else:
            # Fallback for other formats
            req_list = [requirements] if requirements else []
            section_info = 'Unknown'
            logger.info(f"Checking compliance for {len(req_list)} requirements (fallback format)")
    
        results = []
    
        for req in req_list:
            try:
                compliance_result = self._check_single_requirement(bank_policy, req)
                results.append(compliance_result)
            except Exception as e:
                logger.error(f"Error checking single requirement: {str(e)}")
                # Add error result instead of failing completely
                error_result = {
                    "requirement_id": getattr(req, 'get', lambda x, y=None: f"error_req_{len(results)}")('id', f"error_req_{len(results)}"),
                    "compliance_status": "error",
                    "explanation": f"Error processing requirement: {str(e)}",
                    "confidence": 0.0,
                    "recommendations": ["Review requirement format and try again"]
                }
                results.append(error_result)
                continue
    
        return {
            "section_info": section_info,
            "requirements": results,
            "total_checked": len(results)
        }

    
    def _check_single_requirement(self, bank_policy: str, requirement) -> Dict[str, Any]:
        """Check compliance for a single requirement."""
    
        # Handle different requirement formats
        if isinstance(requirement, dict):
            req_id = requirement.get('id', 'N/A')
            req_text = requirement.get('requirement', 'N/A')
            req_reference = requirement.get('reference', 'N/A')
            req_keywords = requirement.get('keywords', [])
        elif isinstance(requirement, str):
            req_id = 'text_requirement'
            req_text = requirement
            req_reference = 'N/A'
            req_keywords = []
        elif isinstance(requirement, list):
            req_id = 'list_requirement'
            req_text = ' '.join(str(item) for item in requirement if item)
            req_reference = 'N/A'
            req_keywords = []
        else:
            req_id = 'unknown_requirement'
            req_text = str(requirement)
            req_reference = 'N/A'
            req_keywords = []
    
        # Skip empty requirements
        if not req_text or req_text.strip() == 'N/A':
            return {
                "id": req_id,
                "requirement": req_text,
                "reference": req_reference,
                "compliance_status": "skipped",
                "assessment": "Skipped empty or invalid requirement",
                "recommendations": [],
                "policy_references": []
            }
    
        prompt = f"""
        Assess the compliance of the bank policy against the following regulatory requirement:
    
        Requirement ID: {req_id}
        Requirement: {req_text}
        Reference: {req_reference}
        Keywords: {req_keywords}
    
        Bank Policy (full text):
        {bank_policy}
    
        Classify the compliance level as one of:
        1. fully_compliant - Requirement is fully addressed
        2. satisfactory - Requirement is adequately addressed with minor gaps
        3. major_gaps - Requirement is partially addressed but has significant gaps
        4. non_compliant - Requirement is addressed but not adequately
        5. missing_requirement - Requirement is not addressed at all
    
        Provide your assessment in the following format:
        Classification: [your classification]
        Reasoning: [detailed explanation]
        Recommendations: [specific recommendations if not fully compliant]
        Policy References: [relevant sections of bank policy that relate to this requirement]
        """
    
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a banking compliance expert. Assess policy compliance accurately and provide actionable recommendations."),
                HumanMessage(content=prompt)
            ])
        
            # Parse response
            classification = self._extract_classification(response.content)
        
            return {
                "id": req_id,
                "requirement": req_text,
                "reference": req_reference,
                "compliance_status": classification,
                "assessment": response.content,
                "recommendations": self._extract_recommendations(response.content),
                "policy_references": self._extract_policy_references(response.content)
            }
        
        except Exception as e:
            logger.error(f"Error checking requirement {req_id}: {str(e)}")
            return {
                "id": req_id,
                "requirement": req_text,
                "reference": req_reference,
                "compliance_status": "error",
                "assessment": f"Error during compliance check: {str(e)}",
                "recommendations": ["Manual review required due to processing error"],
                "policy_references": []
            }
   
    def _extract_classification(self, response: str) -> str:
        """Extract classification from response."""
        classifications = [
            'fully_compliant', 'satisfactory', 'major_gaps', 
            'non_compliant', 'missing_requirement'
        ]
        
        response_lower = response.lower()
        for classification in classifications:
            if classification.replace('_', ' ') in response_lower or classification in response_lower:
                return classification
        
        return 'manual_review_required'
    
    def _extract_recommendations(self, response: str) -> str:
        """Extract recommendations from response."""
        lines = response.split('\n')
        recommendations = []
        capture = False
        
        for line in lines:
            if 'recommendation' in line.lower():
                capture = True
                continue
            if capture and line.strip():
                if any(keyword in line.lower() for keyword in ['classification:', 'reasoning:', 'policy references:']):
                    break
                recommendations.append(line.strip())
        
        return ' '.join(recommendations) if recommendations else "No specific recommendations provided."
    
    def _extract_policy_references(self, response: str) -> str:
        """Extract policy references from response."""
        lines = response.split('\n')
        references = []
        capture = False
        
        for line in lines:
            if 'policy reference' in line.lower():
                capture = True
                continue
            if capture and line.strip():
                references.append(line.strip())
        
        return ' '.join(references) if references else "No specific policy references identified."
