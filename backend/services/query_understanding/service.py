"""
Query Understanding Service

Provides query analysis capabilities before retrieval:
- Query Classification: Categorize query type (definition, how-to, comparison, etc.)
- Intent Detection: Understand user's real intent
- Query Rewriting: Normalize and optimize queries for better retrieval
- Query Expansion: Add related terms for better recall
- Language Detection: Detect query language
- Entity Extraction: Extract key entities from query

This layer improves accuracy by ensuring the right retrieval strategy is used.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================

class QueryType(str, Enum):
    """Types of queries."""
    DEFINITION = "definition"       # What is X?
    HOW_TO = "how_to"               # How to do X?
    COMPARISON = "comparison"       # X vs Y, difference between
    FACTUAL = "factual"             # Specific fact lookup
    OPINION = "opinion"             # What do you think about X?
    CALCULATION = "calculation"     # Math/computation queries
    CODE = "code"                   # Programming related
    POLICY = "policy"               # Policy/rule queries
    TROUBLESHOOTING = "troubleshooting"  # Problem solving
    LIST = "list"                   # Give me a list of X
    TEMPORAL = "temporal"           # Time-based queries
    GENERAL = "general"             # General/other


class Intent(str, Enum):
    """User intents."""
    INFORMATION_SEEKING = "information_seeking"
    TASK_COMPLETION = "task_completion"
    TROUBLESHOOTING = "troubleshooting"
    EXPLORATION = "exploration"
    VERIFICATION = "verification"
    COMPARISON = "comparison"
    DECISION_SUPPORT = "decision_support"


class RetrievalStrategy(str, Enum):
    """Recommended retrieval strategies."""
    SEMANTIC = "semantic"           # Vector similarity
    KEYWORD = "keyword"             # BM25/keyword match
    HYBRID = "hybrid"               # Combined semantic + keyword
    EXACT = "exact"                 # Exact phrase matching
    STRUCTURED = "structured"       # Database/structured query
    WEB = "web"                     # Web search needed
    MEMORY = "memory"               # User memory/history


@dataclass
class Entity:
    """Extracted entity from query."""
    text: str
    entity_type: str  # person, organization, concept, product, etc.
    confidence: float = 1.0
    normalized: Optional[str] = None


@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    original_query: str
    normalized_query: str
    
    # Classification
    query_type: QueryType
    query_type_confidence: float
    
    # Intent
    intent: Intent
    intent_confidence: float
    
    # Entities
    entities: List[Entity] = field(default_factory=list)
    
    # Keywords
    keywords: List[str] = field(default_factory=list)
    
    # Language
    language: str = "en"
    
    # Complexity
    complexity: str = "simple"  # simple, moderate, complex
    
    # Suggested strategies
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    requires_web_search: bool = False
    requires_calculation: bool = False
    requires_code_execution: bool = False
    
    # Rewritten queries
    rewritten_queries: List[str] = field(default_factory=list)
    
    # Expansion terms
    expansion_terms: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Query Classifier
# =============================================================================

class QueryClassifier:
    """
    Classifies queries into types using pattern matching and LLM.
    """
    
    # Pattern-based classification rules
    PATTERNS = {
        QueryType.DEFINITION: [
            r'^what\s+is\s+',
            r'^what\s+are\s+',
            r'^define\s+',
            r'^meaning\s+of\s+',
            r'^explain\s+what\s+',
        ],
        QueryType.HOW_TO: [
            r'^how\s+(?:do|can|to|should)\s+',
            r'^how\s+(?:does|is)\s+\w+\s+(?:work|done)',
            r'^steps\s+to\s+',
            r'^guide\s+(?:to|for)\s+',
            r'^tutorial\s+',
        ],
        QueryType.COMPARISON: [
            r'\bvs\.?\s+',
            r'\bversus\s+',
            r'\bcompare\s+',
            r'\bdifference\s+between\s+',
            r'\bcomparison\s+',
            r'\bor\b.*\bor\b',  # X or Y or Z
        ],
        QueryType.LIST: [
            r'^list\s+',
            r'^what\s+are\s+(?:the|some)\s+',
            r'^give\s+me\s+',
            r'^show\s+(?:me\s+)?(?:all|some)\s+',
            r'^(?:\d+|top|best)\s+\w+',
        ],
        QueryType.TROUBLESHOOTING: [
            r'\berror\b',
            r'\bproblem\b',
            r'\bissue\b',
            r'\bfix\b',
            r'\bsolve\b',
            r'\bnot\s+working\b',
            r'\bfailed\b',
            r'\bhelp\b.*\bwith\b',
        ],
        QueryType.CALCULATION: [
            r'\bcalculate\b',
            r'\bcompute\b',
            r'\bsum\b',
            r'\btotal\b',
            r'\baverage\b',
            r'[\d+\-*/^%]',  # Math operators
        ],
        QueryType.CODE: [
            r'\bcode\b',
            r'\bscript\b',
            r'\bprogram\b',
            r'\bfunction\b',
            r'\bapi\b',
            r'\bpython\b',
            r'\bjavascript\b',
            r'\bsql\b',
        ],
        QueryType.POLICY: [
            r'\bpolicy\b',
            r'\brule\b',
            r'\bguideline\b',
            r'\bregulation\b',
            r'\bcompliance\b',
            r'\ballowed\b',
            r'\bprohibited\b',
        ],
        QueryType.TEMPORAL: [
            r'\bwhen\b',
            r'\bdate\b',
            r'\btime\b',
            r'\bschedule\b',
            r'\bdeadline\b',
            r'\bhistory\b',
        ],
    }
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        # Compile patterns
        self._compiled_patterns = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self.PATTERNS.items()
        }
    
    async def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify the query type.
        
        Returns:
            Tuple of (QueryType, confidence)
        """
        query_lower = query.lower().strip()
        
        # Pattern-based classification
        pattern_result = self._classify_by_pattern(query_lower)
        
        if pattern_result[1] > 0.8:
            return pattern_result
        
        # LLM-based classification if available and pattern uncertain
        if self.llm_client and pattern_result[1] < 0.6:
            try:
                llm_result = await self._classify_by_llm(query)
                if llm_result[1] > pattern_result[1]:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        
        return pattern_result if pattern_result[1] > 0 else (QueryType.GENERAL, 0.5)
    
    def _classify_by_pattern(self, query: str) -> Tuple[QueryType, float]:
        """Classify using pattern matching."""
        scores = {}
        
        for qtype, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[qtype] = scores.get(qtype, 0) + 1
        
        if scores:
            best_type = max(scores, key=scores.get)
            # Normalize confidence
            confidence = min(scores[best_type] / 3.0, 1.0)
            return best_type, confidence
        
        return QueryType.GENERAL, 0.3
    
    async def _classify_by_llm(self, query: str) -> Tuple[QueryType, float]:
        """Classify using LLM."""
        prompt = f"""Classify this query into one category:
- definition (What is X?)
- how_to (How to do X?)
- comparison (X vs Y)
- factual (Specific fact)
- calculation (Math/computation)
- code (Programming)
- policy (Rules/guidelines)
- troubleshooting (Problem solving)
- list (List of items)
- temporal (Time-related)
- general (Other)

Query: {query}

Respond with just the category name."""

        response = await self.llm_client.generate(prompt)
        response = response.strip().lower()
        
        # Map response to QueryType
        for qtype in QueryType:
            if qtype.value in response:
                return qtype, 0.85
        
        return QueryType.GENERAL, 0.5


# =============================================================================
# Intent Detector
# =============================================================================

class IntentDetector:
    """
    Detects user intent from query.
    """
    
    INTENT_PATTERNS = {
        Intent.INFORMATION_SEEKING: [
            r'^what\s+',
            r'^who\s+',
            r'^where\s+',
            r'^when\s+',
            r'^tell\s+me\s+',
            r'\?$',
        ],
        Intent.TASK_COMPLETION: [
            r'^create\s+',
            r'^make\s+',
            r'^build\s+',
            r'^generate\s+',
            r'^write\s+',
            r'^send\s+',
        ],
        Intent.TROUBLESHOOTING: [
            r'\bfix\b',
            r'\bsolve\b',
            r'\bhelp\b',
            r'\berror\b',
            r'\bproblem\b',
            r'\bnot\s+working\b',
        ],
        Intent.COMPARISON: [
            r'\bvs\b',
            r'\bcompare\b',
            r'\bbetter\b',
            r'\bworse\b',
            r'\bdifference\b',
        ],
        Intent.VERIFICATION: [
            r'^is\s+it\s+true\b',
            r'^can\s+you\s+confirm\b',
            r'\bverify\b',
            r'\bcheck\b',
            r'^does\s+',
        ],
        Intent.EXPLORATION: [
            r'\bexplore\b',
            r'\blearn\b',
            r'\bunderstand\b',
            r'^tell\s+me\s+about\b',
        ],
    }
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
    
    async def detect(self, query: str) -> Tuple[Intent, float]:
        """Detect user intent."""
        query_lower = query.lower().strip()
        
        scores = {}
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    scores[intent] = scores.get(intent, 0) + 1
        
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] / 2.0, 1.0)
            return best_intent, confidence
        
        return Intent.INFORMATION_SEEKING, 0.5


# =============================================================================
# Query Rewriter
# =============================================================================

class QueryRewriter:
    """
    Rewrites and optimizes queries for better retrieval.
    """
    
    def __init__(self, llm_client: Any = None, max_rewrites: int = 3):
        self.llm_client = llm_client
        self.max_rewrites = max_rewrites
    
    async def rewrite(
        self,
        query: str,
        query_type: QueryType,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Generate rewritten versions of the query.
        
        Returns list of rewritten queries.
        """
        rewrites = []
        
        # Basic normalization
        normalized = self._normalize(query)
        if normalized != query:
            rewrites.append(normalized)
        
        # Type-specific rewrites
        type_rewrites = self._type_specific_rewrite(query, query_type)
        rewrites.extend(type_rewrites)
        
        # LLM-based rewriting
        if self.llm_client:
            try:
                llm_rewrites = await self._llm_rewrite(query, context)
                rewrites.extend(llm_rewrites)
            except Exception as e:
                logger.warning(f"LLM rewriting failed: {e}")
        
        # Deduplicate and limit
        unique = []
        seen = set()
        for r in rewrites:
            r_lower = r.lower().strip()
            if r_lower not in seen and r_lower != query.lower().strip():
                seen.add(r_lower)
                unique.append(r)
        
        return unique[:self.max_rewrites]
    
    def _normalize(self, query: str) -> str:
        """Normalize query text."""
        # Remove extra whitespace
        normalized = ' '.join(query.split())
        
        # Remove trailing question mark for processing
        # (will be handled by classification)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\binfo\b': 'information',
            r'\bdocs?\b': 'documentation',
            r'\bconfig\b': 'configuration',
        }
        for pattern, replacement in abbreviations.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _type_specific_rewrite(
        self,
        query: str,
        query_type: QueryType
    ) -> List[str]:
        """Generate rewrites based on query type."""
        rewrites = []
        
        if query_type == QueryType.DEFINITION:
            # Add "definition of" prefix
            core = re.sub(r'^what\s+is\s+(?:a\s+|an\s+|the\s+)?', '', query, flags=re.IGNORECASE)
            rewrites.append(f"definition of {core}")
            rewrites.append(f"{core} meaning")
        
        elif query_type == QueryType.HOW_TO:
            # Convert to various how-to formats
            core = re.sub(r'^how\s+(?:to|do|can|should)\s+(?:I\s+)?', '', query, flags=re.IGNORECASE)
            rewrites.append(f"steps to {core}")
            rewrites.append(f"guide for {core}")
        
        elif query_type == QueryType.TROUBLESHOOTING:
            # Add solution-focused variants
            rewrites.append(f"solution for {query}")
            rewrites.append(f"fix for {query}")
        
        elif query_type == QueryType.COMPARISON:
            # Extract entities and create explicit comparison
            entities = re.findall(r'(\w+)\s+(?:vs|versus|or|compared\s+to)\s+(\w+)', query, re.IGNORECASE)
            if entities:
                e1, e2 = entities[0]
                rewrites.append(f"differences between {e1} and {e2}")
                rewrites.append(f"{e1} versus {e2} comparison")
        
        return rewrites
    
    async def _llm_rewrite(
        self,
        query: str,
        context: Optional[str] = None
    ) -> List[str]:
        """Generate rewrites using LLM."""
        context_str = f"\nContext: {context}" if context else ""
        
        prompt = f"""Rewrite this search query to improve retrieval. Generate 2 alternative versions that:
1. Are more specific
2. Use different keywords
3. Remove ambiguity

Original query: {query}{context_str}

Return only the rewritten queries, one per line."""

        response = await self.llm_client.generate(prompt)
        
        # Parse lines
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        # Remove numbering
        lines = [re.sub(r'^\d+[\.\)]\s*', '', l) for l in lines]
        
        return lines[:2]


# =============================================================================
# Query Expander
# =============================================================================

class QueryExpander:
    """
    Expands queries with related terms for better recall.
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
    
    async def expand(
        self,
        query: str,
        max_terms: int = 5
    ) -> List[str]:
        """
        Generate expansion terms for the query.
        
        Returns list of related terms.
        """
        terms = []
        
        # LLM-based expansion
        if self.llm_client:
            try:
                prompt = f"""Generate {max_terms} related keywords/terms for this search query.
Return only the terms, comma-separated.

Query: {query}

Related terms:"""

                response = await self.llm_client.generate(prompt)
                
                # Parse comma-separated terms
                extracted = [t.strip() for t in response.split(',')]
                terms.extend(extracted[:max_terms])
            except Exception as e:
                logger.warning(f"LLM expansion failed: {e}")
        
        return terms


# =============================================================================
# Entity Extractor
# =============================================================================

class EntityExtractor:
    """
    Extracts named entities from queries.
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
    
    async def extract(self, query: str) -> List[Entity]:
        """Extract entities from the query."""
        entities = []
        
        # Pattern-based extraction
        patterns = {
            'quoted': r'"([^"]+)"',  # Quoted phrases
            'capitalized': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Multi-word proper nouns
            'acronym': r'\b([A-Z]{2,})\b',  # Acronyms
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query)
            for match in matches:
                entities.append(Entity(
                    text=match,
                    entity_type=entity_type,
                    confidence=0.8
                ))
        
        return entities


# =============================================================================
# Query Understanding Service
# =============================================================================

class QueryUnderstandingService:
    """
    Unified query understanding service combining all capabilities.
    """
    
    def __init__(
        self,
        llm_client: Any = None,
        enable_classification: bool = True,
        enable_intent: bool = True,
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_entity_extraction: bool = True
    ):
        self.llm_client = llm_client
        
        self.classifier = QueryClassifier(llm_client) if enable_classification else None
        self.intent_detector = IntentDetector(llm_client) if enable_intent else None
        self.rewriter = QueryRewriter(llm_client) if enable_rewriting else None
        self.expander = QueryExpander(llm_client) if enable_expansion else None
        self.entity_extractor = EntityExtractor(llm_client) if enable_entity_extraction else None
    
    async def analyze(
        self,
        query: str,
        context: Optional[str] = None
    ) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: The user query
            context: Optional conversation context
            
        Returns:
            Complete QueryAnalysis
        """
        # Initialize result
        analysis = QueryAnalysis(
            original_query=query,
            normalized_query=query,
            query_type=QueryType.GENERAL,
            query_type_confidence=0.5,
            intent=Intent.INFORMATION_SEEKING,
            intent_confidence=0.5
        )
        
        # Run analysis tasks in parallel
        tasks = []
        
        if self.classifier:
            tasks.append(('classify', self.classifier.classify(query)))
        
        if self.intent_detector:
            tasks.append(('intent', self.intent_detector.detect(query)))
        
        if self.entity_extractor:
            tasks.append(('entities', self.entity_extractor.extract(query)))
        
        # Execute tasks
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )
            for (name, _), result in zip(tasks, task_results):
                if not isinstance(result, Exception):
                    results[name] = result
        
        # Process classification
        if 'classify' in results:
            analysis.query_type, analysis.query_type_confidence = results['classify']
        
        # Process intent
        if 'intent' in results:
            analysis.intent, analysis.intent_confidence = results['intent']
        
        # Process entities
        if 'entities' in results:
            analysis.entities = results['entities']
        
        # Run rewriting (depends on classification)
        if self.rewriter:
            analysis.rewritten_queries = await self.rewriter.rewrite(
                query, analysis.query_type, context
            )
        
        # Run expansion
        if self.expander:
            analysis.expansion_terms = await self.expander.expand(query)
        
        # Normalize query
        analysis.normalized_query = self._normalize_query(query)
        
        # Extract keywords
        analysis.keywords = self._extract_keywords(query)
        
        # Determine complexity
        analysis.complexity = self._assess_complexity(query, analysis)
        
        # Determine retrieval strategy
        analysis.retrieval_strategy = self._determine_strategy(analysis)
        
        # Check special requirements
        analysis.requires_calculation = analysis.query_type == QueryType.CALCULATION
        analysis.requires_code_execution = analysis.query_type == QueryType.CODE
        analysis.requires_web_search = self._needs_web_search(query, analysis)
        
        logger.info(
            f"Query analysis: type={analysis.query_type.value}, "
            f"intent={analysis.intent.value}, "
            f"strategy={analysis.retrieval_strategy.value}"
        )
        
        return analysis
    
    def _normalize_query(self, query: str) -> str:
        """Normalize the query."""
        normalized = ' '.join(query.split())
        return normalized
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Remove stopwords and punctuation
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'what', 'which', 'who',
            'how', 'when', 'where', 'why', 'to', 'for', 'of', 'in', 'on',
            'at', 'by', 'with', 'from', 'as', 'into', 'and', 'or', 'but',
            'if', 'then', 'else', 'so', 'because', 'about', 'between',
            'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'their',
            'it', 'its', 'please', 'help'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _assess_complexity(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> str:
        """Assess query complexity."""
        word_count = len(query.split())
        
        if word_count < 8 and len(analysis.entities) <= 1:
            return "simple"
        elif word_count < 20 or len(analysis.entities) <= 2:
            return "moderate"
        else:
            return "complex"
    
    def _determine_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Determine optimal retrieval strategy."""
        if analysis.query_type == QueryType.CALCULATION:
            return RetrievalStrategy.STRUCTURED
        
        if analysis.query_type in [QueryType.DEFINITION, QueryType.FACTUAL]:
            return RetrievalStrategy.SEMANTIC
        
        if analysis.query_type == QueryType.POLICY:
            return RetrievalStrategy.EXACT
        
        if analysis.query_type == QueryType.TROUBLESHOOTING:
            return RetrievalStrategy.HYBRID
        
        return RetrievalStrategy.HYBRID
    
    def _needs_web_search(self, query: str, analysis: QueryAnalysis) -> bool:
        """Check if query needs web search."""
        web_indicators = [
            'current', 'latest', 'recent', 'today', 'news',
            'price', 'weather', 'stock', 'trending'
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in web_indicators)


# =============================================================================
# Factory Functions
# =============================================================================

_query_service: Optional[QueryUnderstandingService] = None


def get_query_understanding_service(
    llm_client: Any = None,
    **kwargs
) -> QueryUnderstandingService:
    """Get or create the global query understanding service."""
    global _query_service
    
    if _query_service is None:
        _query_service = QueryUnderstandingService(
            llm_client=llm_client,
            **kwargs
        )
    
    return _query_service


def create_query_understanding_service(
    llm_client: Any = None,
    **kwargs
) -> QueryUnderstandingService:
    """Create a new query understanding service."""
    return QueryUnderstandingService(llm_client=llm_client, **kwargs)
