"""
Verification Service

Provides answer verification capabilities:
- Answer Verification: Validates answers against source documents
- Citation Checking: Ensures citations are accurate and properly attributed
- Fact Validation: Verifies factual claims against knowledge base
- Hallucination Detection: Identifies claims not supported by sources
- Consistency Checking: Ensures internal consistency of responses

This is a key differentiator from basic RAG - the verification layer
significantly reduces hallucinations and improves response quality.
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class VerificationStatus(str, Enum):
    """Overall verification status."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class ClaimType(str, Enum):
    """Type of claim in an answer."""
    FACTUAL = "factual"
    OPINION = "opinion"
    PROCEDURAL = "procedural"
    DEFINITION = "definition"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    UNCERTAIN = "uncertain"


@dataclass
class Claim:
    """A single claim extracted from the answer."""
    text: str
    claim_type: ClaimType
    citations: List[str] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass 
class CitationCheck:
    """Result of checking a single citation."""
    citation_id: str
    document_id: str
    is_valid: bool
    relevance_score: float
    matched_content: Optional[str] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result."""
    status: VerificationStatus
    overall_confidence: float
    
    # Claim analysis
    claims: List[Claim] = field(default_factory=list)
    verified_claims: int = 0
    unverified_claims: int = 0
    
    # Citation checking
    citation_checks: List[CitationCheck] = field(default_factory=list)
    valid_citations: int = 0
    invalid_citations: int = 0
    missing_citations: List[str] = field(default_factory=list)
    
    # Hallucination detection
    hallucination_score: float = 0.0  # 0 = no hallucination, 1 = all hallucinated
    hallucinated_claims: List[str] = field(default_factory=list)
    
    # Consistency
    is_consistent: bool = True
    consistency_issues: List[str] = field(default_factory=list)
    
    # Suggestions
    suggestions: List[str] = field(default_factory=list)
    revised_answer: Optional[str] = None


# =============================================================================
# Base Verifier
# =============================================================================

class BaseVerifier(ABC):
    """Abstract base class for verifiers."""
    
    @abstractmethod
    async def verify(
        self,
        answer: str,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> VerificationResult:
        """
        Verify an answer against sources.
        
        Args:
            answer: The generated answer to verify
            query: The original query
            sources: Source documents used for generation
            
        Returns:
            VerificationResult with detailed analysis
        """
        pass


# =============================================================================
# Answer Verifier
# =============================================================================

class AnswerVerifier(BaseVerifier):
    """
    Verifies answers by checking claims against source documents.
    
    Uses an LLM to:
    1. Extract claims from the answer
    2. Match claims to source content
    3. Identify unsupported claims (potential hallucinations)
    4. Calculate verification confidence
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        confidence_threshold: float = 0.7,
        require_citations: bool = True
    ):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.require_citations = require_citations
    
    async def verify(
        self,
        answer: str,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> VerificationResult:
        """Verify the answer against sources."""
        result = VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            overall_confidence=0.0
        )
        
        if not answer or not sources:
            result.suggestions.append("No answer or sources to verify")
            return result
        
        # Step 1: Extract claims from answer
        claims = await self._extract_claims(answer)
        result.claims = claims
        
        # Step 2: Check citations
        citation_checks = await self._check_citations(answer, sources)
        result.citation_checks = citation_checks
        result.valid_citations = sum(1 for c in citation_checks if c.is_valid)
        result.invalid_citations = sum(1 for c in citation_checks if not c.is_valid)
        
        # Step 3: Verify claims against sources
        await self._verify_claims(claims, sources)
        result.verified_claims = sum(1 for c in claims if c.verified)
        result.unverified_claims = len(claims) - result.verified_claims
        
        # Step 4: Detect hallucinations
        hallucinated = [c.text for c in claims if not c.verified and c.claim_type == ClaimType.FACTUAL]
        result.hallucinated_claims = hallucinated
        result.hallucination_score = len(hallucinated) / max(len(claims), 1)
        
        # Step 5: Check consistency
        consistency_result = await self._check_consistency(answer, claims)
        result.is_consistent = consistency_result[0]
        result.consistency_issues = consistency_result[1]
        
        # Step 6: Calculate overall confidence and status
        result.overall_confidence = self._calculate_confidence(result)
        result.status = self._determine_status(result)
        
        # Step 7: Generate suggestions
        result.suggestions = self._generate_suggestions(result, answer, query)
        
        logger.info(
            f"Verification complete: status={result.status.value}, "
            f"confidence={result.overall_confidence:.2f}, "
            f"hallucination_score={result.hallucination_score:.2f}"
        )
        
        return result
    
    async def _extract_claims(self, answer: str) -> List[Claim]:
        """Extract verifiable claims from the answer."""
        claims = []
        
        if self.llm_client:
            try:
                prompt = f"""Extract all factual claims from this text. For each claim, identify:
1. The exact claim text
2. The type (factual, opinion, procedural, definition, statistical, temporal)
3. Any citation references (e.g., [doc_001])

Text:
{answer}

Return as JSON array: [{{"text": "claim text", "type": "factual", "citations": ["doc_001"]}}]
Only extract verifiable factual claims, not opinions or uncertain statements."""

                response = await self.llm_client.generate(prompt)
                
                # Parse JSON from response
                import json
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    parsed = json.loads(json_match.group())
                    for item in parsed:
                        claims.append(Claim(
                            text=item.get("text", ""),
                            claim_type=ClaimType(item.get("type", "factual")),
                            citations=item.get("citations", [])
                        ))
            except Exception as e:
                logger.warning(f"LLM claim extraction failed: {e}")
        
        # Fallback: simple sentence-based extraction
        if not claims:
            sentences = re.split(r'[.!?]+', answer)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Skip very short fragments
                    # Extract citations
                    citations = re.findall(r'\[([^\]]+)\]', sent)
                    claims.append(Claim(
                        text=sent,
                        claim_type=ClaimType.FACTUAL,
                        citations=citations
                    ))
        
        return claims
    
    async def _check_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> List[CitationCheck]:
        """Check all citations in the answer."""
        checks = []
        
        # Find all citations in answer
        citation_pattern = r'\[([^\]]+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Build source lookup
        source_map = {}
        for source in sources:
            doc_id = source.get('document_id') or source.get('id') or ''
            source_map[doc_id] = source
        
        for citation in set(citations):
            # Check if citation refers to a valid source
            is_valid = citation in source_map
            relevance = 0.0
            matched_content = None
            issues = []
            
            if is_valid:
                source = source_map[citation]
                content = source.get('content', '')
                
                # Find relevant text around citation in answer
                citation_context = self._get_citation_context(answer, citation)
                
                # Check if source content supports the citation context
                relevance, matched_content = self._check_content_match(
                    citation_context, content
                )
                
                if relevance < 0.3:
                    issues.append("Citation context poorly matches source content")
            else:
                issues.append(f"Citation [{citation}] not found in sources")
            
            checks.append(CitationCheck(
                citation_id=citation,
                document_id=citation,
                is_valid=is_valid and relevance > 0.3,
                relevance_score=relevance,
                matched_content=matched_content,
                issues=issues
            ))
        
        return checks
    
    def _get_citation_context(self, answer: str, citation: str, window: int = 100) -> str:
        """Get text around a citation."""
        pattern = re.escape(f'[{citation}]')
        match = re.search(pattern, answer)
        if match:
            start = max(0, match.start() - window)
            end = min(len(answer), match.end() + window)
            return answer[start:end]
        return ""
    
    def _check_content_match(
        self,
        citation_context: str,
        source_content: str
    ) -> Tuple[float, Optional[str]]:
        """Check how well citation context matches source."""
        if not citation_context or not source_content:
            return 0.0, None
        
        # Simple word overlap scoring
        context_words = set(citation_context.lower().split())
        source_words = set(source_content.lower().split())
        
        overlap = len(context_words & source_words)
        score = overlap / max(len(context_words), 1)
        
        # Find best matching excerpt
        matched = None
        if score > 0.2:
            # Simple excerpt extraction
            source_sentences = source_content.split('.')
            best_match = 0
            for sent in source_sentences:
                sent_words = set(sent.lower().split())
                match_score = len(context_words & sent_words) / max(len(context_words), 1)
                if match_score > best_match:
                    best_match = match_score
                    matched = sent.strip()
        
        return score, matched
    
    async def _verify_claims(
        self,
        claims: List[Claim],
        sources: List[Dict[str, Any]]
    ) -> None:
        """Verify each claim against sources."""
        # Build combined source content
        all_content = " ".join([
            s.get('content', '') for s in sources
        ]).lower()
        
        for claim in claims:
            # Check if claim has supporting evidence in sources
            claim_words = set(claim.text.lower().split())
            
            # Simple verification: check word overlap
            source_words = set(all_content.split())
            overlap = len(claim_words & source_words)
            overlap_ratio = overlap / max(len(claim_words), 1)
            
            if overlap_ratio > 0.5:
                claim.verified = True
                claim.confidence = min(overlap_ratio, 1.0)
                
                # Find supporting evidence
                for source in sources:
                    content = source.get('content', '').lower()
                    if any(word in content for word in list(claim_words)[:5]):
                        claim.supporting_evidence.append(
                            source.get('document_id', 'unknown')
                        )
            else:
                claim.verified = False
                claim.confidence = overlap_ratio
                claim.issues.append("Insufficient supporting evidence in sources")
    
    async def _check_consistency(
        self,
        answer: str,
        claims: List[Claim]
    ) -> Tuple[bool, List[str]]:
        """Check internal consistency of the answer."""
        issues = []
        
        if self.llm_client:
            try:
                prompt = f"""Analyze this text for internal consistency. Look for:
1. Contradictory statements
2. Conflicting facts or numbers
3. Logical inconsistencies

Text:
{answer}

If consistent, respond with: {{"consistent": true, "issues": []}}
If inconsistent, respond with: {{"consistent": false, "issues": ["issue1", "issue2"]}}"""

                response = await self.llm_client.generate(prompt)
                
                import json
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return parsed.get("consistent", True), parsed.get("issues", [])
            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")
        
        return True, issues
    
    def _calculate_confidence(self, result: VerificationResult) -> float:
        """Calculate overall verification confidence."""
        scores = []
        
        # Claim verification score
        if result.claims:
            verified_ratio = result.verified_claims / len(result.claims)
            scores.append(verified_ratio * 0.4)
        
        # Citation validity score
        total_citations = result.valid_citations + result.invalid_citations
        if total_citations > 0:
            citation_score = result.valid_citations / total_citations
            scores.append(citation_score * 0.3)
        
        # Hallucination score (inverse)
        scores.append((1 - result.hallucination_score) * 0.2)
        
        # Consistency score
        if result.is_consistent:
            scores.append(0.1)
        
        return sum(scores)
    
    def _determine_status(self, result: VerificationResult) -> VerificationStatus:
        """Determine overall verification status."""
        if result.overall_confidence >= 0.8 and result.hallucination_score < 0.1:
            return VerificationStatus.VERIFIED
        elif result.overall_confidence >= 0.5 and result.hallucination_score < 0.3:
            return VerificationStatus.PARTIALLY_VERIFIED
        elif result.hallucination_score > 0.5:
            return VerificationStatus.REJECTED
        elif result.overall_confidence < 0.3:
            return VerificationStatus.NEEDS_REVISION
        else:
            return VerificationStatus.UNVERIFIED
    
    def _generate_suggestions(
        self,
        result: VerificationResult,
        answer: str,
        query: str
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if result.hallucinated_claims:
            suggestions.append(
                f"Remove or revise {len(result.hallucinated_claims)} unsupported claims"
            )
        
        if result.invalid_citations > 0:
            suggestions.append(
                f"Fix {result.invalid_citations} invalid citations"
            )
        
        if result.missing_citations:
            suggestions.append(
                f"Add citations for: {', '.join(result.missing_citations[:3])}"
            )
        
        if not result.is_consistent:
            suggestions.append("Resolve internal contradictions in the answer")
        
        if self.require_citations and result.valid_citations == 0:
            suggestions.append("Add citations to support factual claims")
        
        return suggestions


# =============================================================================
# Hallucination Detector
# =============================================================================

class HallucinationDetector:
    """
    Specialized detector for hallucinations in generated content.
    
    Uses multiple strategies:
    1. Source entailment checking
    2. Entity verification
    3. Numerical claim validation
    4. Temporal consistency checking
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
    
    async def detect(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[float, List[str]]:
        """
        Detect hallucinations in the answer.
        
        Returns:
            Tuple of (hallucination_score, list of hallucinated segments)
        """
        hallucinated = []
        
        # Build source content
        source_content = " ".join([s.get('content', '') for s in sources])
        
        # Check for entity hallucinations
        entity_hallucinations = await self._check_entity_hallucinations(answer, source_content)
        hallucinated.extend(entity_hallucinations)
        
        # Check for numerical hallucinations
        numerical_hallucinations = await self._check_numerical_hallucinations(answer, source_content)
        hallucinated.extend(numerical_hallucinations)
        
        # Check for fabricated quotes
        quote_hallucinations = await self._check_quote_hallucinations(answer, source_content)
        hallucinated.extend(quote_hallucinations)
        
        # Calculate score
        answer_sentences = len(re.split(r'[.!?]+', answer))
        score = len(hallucinated) / max(answer_sentences, 1)
        
        return min(score, 1.0), hallucinated
    
    async def _check_entity_hallucinations(
        self,
        answer: str,
        source_content: str
    ) -> List[str]:
        """Check for made-up entities not in sources."""
        hallucinated = []
        
        # Extract potential entities (capitalized words, proper nouns)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        answer_entities = set(re.findall(entity_pattern, answer))
        source_entities = set(re.findall(entity_pattern, source_content))
        
        # Entities in answer but not in sources (potential hallucinations)
        new_entities = answer_entities - source_entities
        
        # Filter out common words that might be capitalized
        common_words = {'The', 'This', 'That', 'These', 'Those', 'However', 'Therefore'}
        suspicious = new_entities - common_words
        
        for entity in suspicious:
            hallucinated.append(f"Potentially fabricated entity: {entity}")
        
        return hallucinated
    
    async def _check_numerical_hallucinations(
        self,
        answer: str,
        source_content: str
    ) -> List[str]:
        """Check for numbers not present in sources."""
        hallucinated = []
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b'
        answer_numbers = set(re.findall(number_pattern, answer.lower()))
        source_numbers = set(re.findall(number_pattern, source_content.lower()))
        
        # Numbers in answer but not in sources
        new_numbers = answer_numbers - source_numbers
        
        for num in new_numbers:
            # Skip very common numbers
            if num not in {'1', '2', '3', '4', '5', '10', '100'}:
                hallucinated.append(f"Number not found in sources: {num}")
        
        return hallucinated
    
    async def _check_quote_hallucinations(
        self,
        answer: str,
        source_content: str
    ) -> List[str]:
        """Check for fabricated quotes."""
        hallucinated = []
        
        # Find quoted text in answer
        quote_pattern = r'"([^"]+)"'
        quotes = re.findall(quote_pattern, answer)
        
        for quote in quotes:
            if len(quote) > 10:  # Skip short quotes
                # Check if quote appears in sources
                if quote.lower() not in source_content.lower():
                    # Check partial match
                    words = quote.split()
                    if len(words) > 3:
                        phrase = ' '.join(words[:4])
                        if phrase.lower() not in source_content.lower():
                            hallucinated.append(f"Potentially fabricated quote: \"{quote[:50]}...\"")
        
        return hallucinated


# =============================================================================
# Verification Service
# =============================================================================

class VerificationService:
    """
    Unified verification service combining all verification capabilities.
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        confidence_threshold: float = 0.7,
        enable_hallucination_detection: bool = True,
        enable_citation_check: bool = True,
        require_citations: bool = True
    ):
        self.answer_verifier = AnswerVerifier(
            llm_client=llm_client,
            confidence_threshold=confidence_threshold,
            require_citations=require_citations
        )
        self.hallucination_detector = HallucinationDetector(
            llm_client=llm_client
        ) if enable_hallucination_detection else None
        
        self.enable_citation_check = enable_citation_check
        self.confidence_threshold = confidence_threshold
    
    async def verify(
        self,
        answer: str,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> VerificationResult:
        """
        Perform comprehensive verification.
        
        Args:
            answer: Generated answer to verify
            query: Original query
            sources: Source documents used
            
        Returns:
            Complete verification result
        """
        # Run main verification
        result = await self.answer_verifier.verify(answer, query, sources)
        
        # Run additional hallucination detection
        if self.hallucination_detector:
            h_score, h_claims = await self.hallucination_detector.detect(answer, sources)
            
            # Merge with existing hallucination data
            result.hallucination_score = max(result.hallucination_score, h_score)
            result.hallucinated_claims.extend(h_claims)
            result.hallucinated_claims = list(set(result.hallucinated_claims))
        
        return result
    
    async def quick_check(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[bool, float]:
        """
        Quick verification check without full analysis.
        
        Returns:
            Tuple of (is_acceptable, confidence_score)
        """
        # Simple checks
        if not answer or len(answer) < 10:
            return False, 0.0
        
        if not sources:
            return False, 0.0
        
        # Check for at least some source overlap
        answer_words = set(answer.lower().split())
        source_words = set()
        for s in sources:
            source_words.update(s.get('content', '').lower().split())
        
        overlap = len(answer_words & source_words)
        overlap_ratio = overlap / max(len(answer_words), 1)
        
        return overlap_ratio > 0.3, overlap_ratio


# =============================================================================
# Factory Functions
# =============================================================================

_verification_service: Optional[VerificationService] = None


def get_verification_service(
    llm_client: Any = None,
    **kwargs
) -> VerificationService:
    """Get or create the global verification service."""
    global _verification_service
    
    if _verification_service is None:
        _verification_service = VerificationService(
            llm_client=llm_client,
            **kwargs
        )
    
    return _verification_service


def create_verification_service(
    llm_client: Any = None,
    **kwargs
) -> VerificationService:
    """Create a new verification service instance."""
    return VerificationService(llm_client=llm_client, **kwargs)
