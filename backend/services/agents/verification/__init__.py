"""
Verification Service Module

Provides answer verification, citation checking, and hallucination detection.
"""

from .service import (
    VerificationService,
    AnswerVerifier,
    HallucinationDetector,
    VerificationResult,
    VerificationStatus,
    Claim,
    ClaimType,
    CitationCheck,
    get_verification_service,
    create_verification_service
)

__all__ = [
    "VerificationService",
    "AnswerVerifier",
    "HallucinationDetector",
    "VerificationResult",
    "VerificationStatus",
    "Claim",
    "ClaimType",
    "CitationCheck",
    "get_verification_service",
    "create_verification_service"
]
