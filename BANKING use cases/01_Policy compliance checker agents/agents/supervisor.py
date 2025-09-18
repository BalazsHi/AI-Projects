from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0.1
        )
    
    def assess_document_length(self, document_text: str, max_tokens: int = 30000) -> Dict[str, Any]:
        """Assess if document needs chunking."""
        estimated_tokens = len(document_text.split()) * 1.3
        
        logger.info(f"Estimated tokens: {estimated_tokens}")
        
        needs_chunking = estimated_tokens > max_tokens
        
        if needs_chunking:
            # Ask LLM to suggest optimal number of chunks
            prompt = f"""
            Analyze this regulatory document and suggest how to divide it into logical sections for processing.
            The document has approximately {estimated_tokens} tokens and needs to be divided into chunks of maximum {max_tokens} tokens each.
            
            Provide your recommendation as a number of suggested chunks and brief reasoning.
            
            Document preview (first 1000 characters):
            {document_text[:1000]}...
            """
            
            response = self.llm.invoke([
                SystemMessage(content="You are a document analysis expert specializing in regulatory documents."),
                HumanMessage(content=prompt)
            ])
            
            # Extract suggested number of chunks (simple parsing)
            suggested_chunks = max(2, int(estimated_tokens / max_tokens) + 1)
            
            return {
                "needs_chunking": True,
                "estimated_tokens": estimated_tokens,
                "suggested_chunks": suggested_chunks,
                "reasoning": response.content
            }
        
        return {
            "needs_chunking": False,
            "estimated_tokens": estimated_tokens,
            "suggested_chunks": 1,
            "reasoning": "Document is within acceptable token limit for single processing."
        }
    
    def create_final_report(self, compliance_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create final compliance report."""
        logger.info("Creating final compliance report...")
        
        # Combine all results
        all_requirements = []
        for result in compliance_results:
            all_requirements.extend(result.get("requirements", []))
        
        # Generate summary statistics
        total_requirements = len(all_requirements)
        compliance_counts = {}
        for req in all_requirements:
            status = req.get("compliance_status", "unknown")
            compliance_counts[status] = compliance_counts.get(status, 0) + 1
        
        # Generate executive summary using LLM
        summary_prompt = f"""
        Create an executive summary for a compliance assessment report with the following statistics:
        - Total requirements assessed: {total_requirements}
        - Compliance breakdown: {compliance_counts}
        
        Provide a professional executive summary highlighting key findings and overall compliance posture.
        """
        
        summary_response = self.llm.invoke([
            SystemMessage(content="You are a compliance expert creating executive summaries for bank regulatory assessments."),
            HumanMessage(content=summary_prompt)
        ])
        
        return {
            "executive_summary": summary_response.content,
            "total_requirements": total_requirements,
            "compliance_breakdown": compliance_counts,
            "detailed_findings": all_requirements,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
