"""Data models for document verification system."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass
from langchain_core.documents import Document


# ==================== SESSION MODELS ====================

class ReferenceData(BaseModel):
    """Reference entities extracted from base document."""

    session_id: str
    entity_type: str
    extraction_query: str
    entities: List[str]
    base_document: str
    source_chunks: List[dict]
    created_at: datetime
    expires_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EntityMatch(BaseModel):
    """Represents a matched entity."""

    reference_entity: str
    target_entity: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_type: Literal["exact", "semantic", "partial", "no_match"]
    explanation: str


class VerificationResult(BaseModel):
    """Results for one target document."""

    target_document: str
    status: Literal["verified", "partial_match", "mismatch"]
    matched_entities: List[EntityMatch]
    missing_in_target: List[str]
    extra_in_target: List[str]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    processed_at: datetime
    extracted_target_entities: List[str] = []  # All entities extracted from target

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ==================== API REQUEST MODELS ====================

class ExtractReferenceRequest(BaseModel):
    """Request to extract reference entities."""

    extraction_query: str = Field(
        ...,
        description="Natural language query describing what to extract",
        examples=["list of employee names", "invoice numbers from Q3 2025"],
        min_length=1
    )
    llm_provider: str = Field(
        "openai",
        description="LLM provider (openai, anthropic)"
    )
    session_ttl: int = Field(
        86400,
        description="Session lifetime in seconds (default: 24 hours)",
        ge=300,
        le=604800  # Max 7 days
    )


class ExtractReferenceResponse(BaseModel):
    """Response with extracted reference data."""

    session_id: str
    entity_type: str
    entities: List[str]
    total_entities: int
    base_document: str
    expires_at: str
    message: str


class CompareTargetRequest(BaseModel):
    """Request to compare target document(s) against reference."""

    session_id: str = Field(
        ...,
        description="Session ID from extract-reference"
    )
    llm_provider: str = Field(
        "openai",
        description="LLM provider"
    )
    strictness: float = Field(
        0.7,
        description="Match strictness threshold",
        ge=0.5,
        le=1.0
    )


class CompareTargetResponse(BaseModel):
    """Response with comparison results."""

    session_id: str
    results: List[dict]  # List of VerificationResult as dicts
    summary_statistics: dict
    message: str


# ==================== INTERNAL DATA STRUCTURES ====================

@dataclass
class ExtractionContext:
    """Context for entity extraction."""

    document_chunks: List[Document]
    query: str
    entity_type: str


@dataclass
class MatchingContext:
    """Context for semantic matching."""

    reference_entities: List[str]
    target_entities: List[str]
    strictness: float
