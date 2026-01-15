"""Core verification engine for document comparison."""

import os
import json
import re
import uuid
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Text normalization for improved matching
try:
    from unidecode import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False
    def unidecode(s):
        return s  # Fallback: return as-is

# Configure logging
logger = logging.getLogger(__name__)

from .document_loader import DocumentLoader
from .chunker import Chunker
from .embeddings import EmbeddingsManager
from .vector_store import VectorStore
from .rag_chain import RAGChain
from .verification_models import (
    ReferenceData,
    EntityMatch,
    VerificationResult,
)
from .verification_session import VerificationSessionManager


class EntityExtractor:
    """Extracts structured entity lists from documents using RAG + LLM."""

    def __init__(self, rag_chain: RAGChain, llm_provider: str = "openai"):
        """
        Initialize entity extractor.

        Args:
            rag_chain: RAGChain instance for retrieval
            llm_provider: LLM provider ("openai" or "anthropic")
        """
        self.rag_chain = rag_chain
        self.llm_provider = llm_provider
        self._llm = self._create_extraction_llm(llm_provider)

    def extract(
        self,
        document_path: str,
        extraction_query: str,
        context_name: str = "verification_temp",
    ) -> Tuple[List[str], List[dict]]:
        """
        Extract entities from document using RAG + LLM.

        Strategy:
        1. Load document into temporary verification context
        2. Use RAG to retrieve relevant chunks
        3. Use LLM with structured output to extract entity list
        4. Return entities + source chunks for provenance

        Args:
            document_path: Path to document file
            extraction_query: Natural language query (e.g., "list of employee names")
            context_name: Temporary context name (will be auto-suffixed with UUID)

        Returns:
            Tuple of (entities list, source chunks list)
        """
        # Load document
        loader = DocumentLoader()
        documents = loader.load(document_path)
        logger.info(f"📄 Loaded {len(documents)} document(s) from {Path(document_path).name}")

        # Chunk documents
        chunker = Chunker.from_config("config.toml")
        chunks = chunker.split(documents)
        logger.info(f"✂️ Split into {len(chunks)} chunks")

        # Index in temporary context with unique suffix
        temp_context = f"{context_name}_{uuid.uuid4().hex[:8]}"
        vector_store = VectorStore(
            embeddings=self.rag_chain.vector_store._embeddings,
            context_name=temp_context,
        )
        vector_store.add_documents(chunks)
        logger.info(f"📊 Indexed {len(chunks)} chunks in FAISS using BGE-M3 embeddings")

        # For entity extraction, we need comprehensive coverage to avoid missing entities
        # Strategy: If document is small (<50 chunks), use ALL chunks
        # If large, use vector search with high top_k to get comprehensive coverage
        max_chunks_for_llm = 50  # Token limit safety: ~25,600 tokens max at 512 chars/chunk

        if len(chunks) <= max_chunks_for_llm:
            # Small document: use all chunks for complete extraction
            relevant_chunks = chunks
            logger.info(f"✅ Using ALL {len(chunks)} chunks for comprehensive extraction")
        else:
            # Large document: use vector search with high top_k
            # This prioritizes most relevant chunks while staying within token limits
            relevant_chunks = vector_store.search_documents(
                query=extraction_query,
                top_k=max_chunks_for_llm,
            )
            logger.info(f"🔍 Using top {len(relevant_chunks)} most relevant chunks (document is large)")

        # Build extraction prompt
        extraction_prompt = self._build_extraction_prompt(
            chunks=relevant_chunks, query=extraction_query
        )

        # Use LLM to extract structured list
        logger.info(f"🤖 Sending {len(relevant_chunks)} chunks to LLM for entity extraction...")
        entities = self._llm_extract_entities(extraction_prompt)
        logger.info(f"✨ Extracted {len(entities)} unique entities: {entities[:5]}{'...' if len(entities) > 5 else ''}")

        # Build source provenance
        source_chunks = [
            {
                "content": chunk.page_content[:200],
                "file": chunk.metadata.get("source", "unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
            }
            for chunk in relevant_chunks[:5]  # Top 5 chunks only
        ]

        # Cleanup temporary context
        try:
            vector_store.delete_index()
        except Exception:
            pass  # Best effort cleanup

        return entities, source_chunks

    def _build_extraction_prompt(
        self, chunks: List[Document], query: str
    ) -> str:
        """Build prompt for entity extraction."""
        context = "\n\n".join(
            [f"[Chunk {i+1}]\n{chunk.page_content}" for i, chunk in enumerate(chunks)]
        )

        return f"""You are an expert at extracting structured information from documents.

TASK: Extract entities based on the user's query.

USER QUERY: "{query}"

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Carefully read all document chunks above
2. Extract ALL entities that match the user's query
3. Extract entities REGARDLESS of formatting - look for the entity itself, not specific prefixes or patterns
4. For names: extract complete person names, ignoring any codes, numbers, or prefixes (e.g., "Colab.:", "40336", etc.)
5. Return ONLY the entities, one per line
6. Do NOT include explanations, numbers, codes, prefixes, or extra text
7. Be consistent with formatting: use proper capitalization (e.g., "João Silva" not "joão silva" or "JOÃO SILVA")
8. Remove duplicate entries
9. If no entities found, return "NONE"

EXAMPLE - If looking for names from these lines:
"Colab.: 40336 IGOR RIBEIRO DA SILVA ok"
"IGOR RIBEIRO DA SILVA"
"40201 MARCOS SANTOS DE SOUZA"

CORRECT OUTPUT:
Igor Ribeiro Da Silva
Marcos Santos De Souza

EXTRACTED ENTITIES (one per line):"""

    def _llm_extract_entities(self, prompt: str) -> List[str]:
        """Use LLM to extract entity list from prompt."""
        response = self._llm.invoke(prompt)

        # Parse response into list
        raw_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        if "NONE" in raw_text.upper():
            return []

        # Split by newlines and clean
        entities = [
            line.strip()
            for line in raw_text.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)

        return unique_entities

    def _create_extraction_llm(self, provider: str):
        """Create LLM instance for extraction."""
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,  # Deterministic extraction
                max_tokens=2000,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.0,
                max_tokens=2000,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


@dataclass
class MatchCandidate:
    """Represents a candidate match with similarity score."""
    ref_entity: str
    ref_index: int
    target_entity: str
    target_index: int
    similarity: float
    normalized_ref: str
    normalized_target: str


class SemanticMatcher:
    """
    Optimized semantic matching between entity lists using three-phase approach:
    1. Exact match (normalized)
    2. High-confidence vector match (>0.95 similarity)
    3. Batch LLM validation for ambiguous cases

    Improvements over original:
    - Batch embedding computation (O(n+m) instead of O(n×m))
    - Text normalization with unidecode
    - Three-phase matching reduces LLM calls by ~80%
    - Cross-validation to prevent duplicate matches
    """

    # Default thresholds (can be overridden via config)
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.95
    DEFAULT_LLM_CANDIDATE_THRESHOLD = 0.70
    DEFAULT_BATCH_SIZE = 10

    def __init__(
        self,
        llm_provider: str = "openai",
        strictness: float = 0.7,
        embeddings=None,
        config_path: str = "config.toml",
    ):
        """
        Initialize semantic matcher.

        Args:
            llm_provider: LLM provider ("openai" or "anthropic")
            strictness: Match threshold (0.5-1.0)
            embeddings: EmbeddingsManager instance for vector similarity
            config_path: Path to config.toml for threshold settings
        """
        self.llm_provider = llm_provider
        self.strictness = strictness
        self.embeddings = embeddings
        self._llm = self._create_matching_llm(llm_provider)

        # Load thresholds from config
        self._load_thresholds_from_config(config_path)

        # Cache for normalized entities
        self._normalization_cache: Dict[str, str] = {}

    def _load_thresholds_from_config(self, config_path: str):
        """Load matching thresholds from config file."""
        try:
            import toml
            config = toml.load(config_path)
            verification_config = config.get("verification", {})

            self.HIGH_CONFIDENCE_THRESHOLD = verification_config.get(
                "high_confidence_threshold", self.DEFAULT_HIGH_CONFIDENCE_THRESHOLD
            )
            self.LLM_CANDIDATE_THRESHOLD = verification_config.get(
                "llm_candidate_threshold", self.DEFAULT_LLM_CANDIDATE_THRESHOLD
            )
            self.BATCH_SIZE = verification_config.get(
                "llm_batch_size", self.DEFAULT_BATCH_SIZE
            )

            logger.info(
                f"⚙️ Matching thresholds: high_conf={self.HIGH_CONFIDENCE_THRESHOLD}, "
                f"llm_candidate={self.LLM_CANDIDATE_THRESHOLD}, batch_size={self.BATCH_SIZE}"
            )
        except Exception as e:
            # Use defaults if config loading fails
            self.HIGH_CONFIDENCE_THRESHOLD = self.DEFAULT_HIGH_CONFIDENCE_THRESHOLD
            self.LLM_CANDIDATE_THRESHOLD = self.DEFAULT_LLM_CANDIDATE_THRESHOLD
            self.BATCH_SIZE = self.DEFAULT_BATCH_SIZE
            logger.warning(f"⚠️ Could not load config, using defaults: {e}")

    def match(
        self, reference_entities: List[str], target_entities: List[str]
    ) -> Tuple[List[EntityMatch], List[str]]:
        """
        Match target entities against reference using optimized three-phase approach.

        Strategy:
        1. Normalize all entities for comparison
        2. Compute similarity matrix in batch
        3. Phase 1: Exact matches (normalized)
        4. Phase 2: High-confidence vector matches (>0.95)
        5. Phase 3: Batch LLM validation for ambiguous cases
        6. Cross-validate to prevent duplicate matches

        Args:
            reference_entities: Reference entity list
            target_entities: Target entity list to match against

        Returns:
            Tuple of (match results, extra entities in target)
        """
        if not reference_entities:
            return [], list(target_entities)

        if not target_entities:
            return [
                EntityMatch(
                    reference_entity=ref,
                    target_entity="",
                    confidence=0.0,
                    match_type="no_match",
                    explanation="No entities in target document",
                )
                for ref in reference_entities
            ], []

        logger.info(f"🎯 Starting optimized matching: {len(reference_entities)} refs × {len(target_entities)} targets")

        # Step 1: Normalize all entities
        ref_normalized = [self._normalize_entity(e) for e in reference_entities]
        target_normalized = [self._normalize_entity(e) for e in target_entities]

        # Create mappings for original <-> normalized
        ref_orig_to_norm = {reference_entities[i]: ref_normalized[i] for i in range(len(reference_entities))}
        target_orig_to_norm = {target_entities[i]: target_normalized[i] for i in range(len(target_entities))}

        # Step 2: Compute similarity matrix in batch
        similarity_matrix = self._compute_similarity_matrix(ref_normalized, target_normalized)

        # Track which entities have been matched
        matched_refs: Set[int] = set()
        matched_targets: Set[int] = set()
        matches: List[EntityMatch] = [None] * len(reference_entities)  # Pre-allocate

        # Phase 1: Exact matches (normalized)
        logger.info("📍 Phase 1: Finding exact matches (normalized)...")
        exact_count = self._phase1_exact_matches(
            reference_entities, target_entities,
            ref_normalized, target_normalized,
            matches, matched_refs, matched_targets
        )
        logger.info(f"   ✅ Found {exact_count} exact matches")

        # Phase 2: High-confidence vector matches
        logger.info(f"📍 Phase 2: Finding high-confidence matches (>{self.HIGH_CONFIDENCE_THRESHOLD})...")
        vector_count = self._phase2_vector_matches(
            reference_entities, target_entities,
            similarity_matrix,
            matches, matched_refs, matched_targets
        )
        logger.info(f"   ✅ Found {vector_count} high-confidence vector matches")

        # Phase 3: Batch LLM validation for remaining candidates
        logger.info(f"📍 Phase 3: LLM validation for ambiguous cases ({self.LLM_CANDIDATE_THRESHOLD} - {self.HIGH_CONFIDENCE_THRESHOLD})...")
        llm_count = self._phase3_llm_validation(
            reference_entities, target_entities,
            similarity_matrix,
            matches, matched_refs, matched_targets
        )
        logger.info(f"   ✅ Found {llm_count} matches via LLM validation")

        # Fill remaining with no_match
        for i, ref in enumerate(reference_entities):
            if matches[i] is None:
                matches[i] = EntityMatch(
                    reference_entity=ref,
                    target_entity="",
                    confidence=0.0,
                    match_type="no_match",
                    explanation="No matching entity found in target",
                )

        # Identify extra entities in target
        extra_entities = [
            target_entities[i] for i in range(len(target_entities))
            if i not in matched_targets
        ]

        total_matched = exact_count + vector_count + llm_count
        logger.info(f"🏁 Matching complete: {total_matched}/{len(reference_entities)} matched, {len(extra_entities)} extra in target")

        return matches, extra_entities

    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize entity for comparison.

        Applies:
        - Strip whitespace
        - Remove accents (unidecode)
        - Lowercase
        - Normalize multiple spaces
        """
        if entity in self._normalization_cache:
            return self._normalization_cache[entity]

        normalized = entity.strip()
        normalized = unidecode(normalized)  # Remove accents
        normalized = normalized.lower()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize spaces

        self._normalization_cache[entity] = normalized
        return normalized

    def _compute_similarity_matrix(
        self, ref_normalized: List[str], target_normalized: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between all refs and targets in batch.

        Returns:
            np.ndarray of shape (n_refs, n_targets) with cosine similarities
        """
        if not self.embeddings:
            # Fallback: return zeros if no embeddings
            return np.zeros((len(ref_normalized), len(target_normalized)))

        try:
            logger.info(f"   Computing embeddings for {len(ref_normalized)} refs + {len(target_normalized)} targets...")

            # Batch embed all entities (much more efficient than one-by-one)
            ref_embeddings = self.embeddings.embeddings.embed_documents(ref_normalized)
            target_embeddings = self.embeddings.embeddings.embed_documents(target_normalized)

            # Convert to numpy arrays
            ref_matrix = np.array(ref_embeddings)
            target_matrix = np.array(target_embeddings)

            # Normalize vectors
            ref_norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
            target_norms = np.linalg.norm(target_matrix, axis=1, keepdims=True)

            ref_normalized_matrix = ref_matrix / np.where(ref_norms > 0, ref_norms, 1)
            target_normalized_matrix = target_matrix / np.where(target_norms > 0, target_norms, 1)

            # Compute similarity matrix via dot product
            similarity_matrix = np.dot(ref_normalized_matrix, target_normalized_matrix.T)

            logger.info(f"   ✅ Similarity matrix computed: {similarity_matrix.shape}")
            return similarity_matrix

        except Exception as e:
            logger.warning(f"   ⚠️ Error computing similarity matrix: {e}")
            return np.zeros((len(ref_normalized), len(target_normalized)))

    def _phase1_exact_matches(
        self,
        reference_entities: List[str],
        target_entities: List[str],
        ref_normalized: List[str],
        target_normalized: List[str],
        matches: List[Optional[EntityMatch]],
        matched_refs: Set[int],
        matched_targets: Set[int],
    ) -> int:
        """Phase 1: Find exact matches using normalized comparison."""
        count = 0

        # Build lookup for target normalized -> indices
        target_lookup: Dict[str, List[int]] = {}
        for i, norm in enumerate(target_normalized):
            if norm not in target_lookup:
                target_lookup[norm] = []
            target_lookup[norm].append(i)

        for ref_idx, ref_norm in enumerate(ref_normalized):
            if ref_idx in matched_refs:
                continue

            if ref_norm in target_lookup:
                # Find first unmatched target with this normalized form
                for target_idx in target_lookup[ref_norm]:
                    if target_idx not in matched_targets:
                        matches[ref_idx] = EntityMatch(
                            reference_entity=reference_entities[ref_idx],
                            target_entity=target_entities[target_idx],
                            confidence=1.0,
                            match_type="exact",
                            explanation="Exact match (normalized)",
                        )
                        matched_refs.add(ref_idx)
                        matched_targets.add(target_idx)
                        count += 1
                        break

        return count

    def _phase2_vector_matches(
        self,
        reference_entities: List[str],
        target_entities: List[str],
        similarity_matrix: np.ndarray,
        matches: List[Optional[EntityMatch]],
        matched_refs: Set[int],
        matched_targets: Set[int],
    ) -> int:
        """Phase 2: Find high-confidence matches based on vector similarity."""
        count = 0

        # Find all pairs above threshold
        candidates = []
        for ref_idx in range(len(reference_entities)):
            if ref_idx in matched_refs:
                continue
            for target_idx in range(len(target_entities)):
                if target_idx in matched_targets:
                    continue
                sim = similarity_matrix[ref_idx, target_idx]
                if sim >= self.HIGH_CONFIDENCE_THRESHOLD:
                    candidates.append((ref_idx, target_idx, sim))

        # Sort by similarity (highest first) and greedily assign
        candidates.sort(key=lambda x: x[2], reverse=True)

        for ref_idx, target_idx, sim in candidates:
            if ref_idx in matched_refs or target_idx in matched_targets:
                continue

            matches[ref_idx] = EntityMatch(
                reference_entity=reference_entities[ref_idx],
                target_entity=target_entities[target_idx],
                confidence=float(sim),
                match_type="semantic",
                explanation=f"High-confidence vector match (similarity: {sim:.3f})",
            )
            matched_refs.add(ref_idx)
            matched_targets.add(target_idx)
            count += 1

        return count

    def _phase3_llm_validation(
        self,
        reference_entities: List[str],
        target_entities: List[str],
        similarity_matrix: np.ndarray,
        matches: List[Optional[EntityMatch]],
        matched_refs: Set[int],
        matched_targets: Set[int],
    ) -> int:
        """Phase 3: Use batch LLM validation for ambiguous candidates."""
        count = 0

        # Collect candidates for LLM validation
        candidates: List[MatchCandidate] = []

        for ref_idx in range(len(reference_entities)):
            if ref_idx in matched_refs:
                continue

            # Get best unmatched targets for this reference
            ref_candidates = []
            for target_idx in range(len(target_entities)):
                if target_idx in matched_targets:
                    continue
                sim = similarity_matrix[ref_idx, target_idx]
                if self.LLM_CANDIDATE_THRESHOLD <= sim < self.HIGH_CONFIDENCE_THRESHOLD:
                    ref_candidates.append((target_idx, sim))

            # Take top 3 candidates for this reference
            ref_candidates.sort(key=lambda x: x[1], reverse=True)
            for target_idx, sim in ref_candidates[:3]:
                candidates.append(MatchCandidate(
                    ref_entity=reference_entities[ref_idx],
                    ref_index=ref_idx,
                    target_entity=target_entities[target_idx],
                    target_index=target_idx,
                    similarity=sim,
                    normalized_ref=self._normalize_entity(reference_entities[ref_idx]),
                    normalized_target=self._normalize_entity(target_entities[target_idx]),
                ))

        if not candidates:
            return 0

        logger.info(f"   Sending {len(candidates)} candidates to LLM for validation...")

        # Process in batches
        for batch_start in range(0, len(candidates), self.BATCH_SIZE):
            batch = candidates[batch_start:batch_start + self.BATCH_SIZE]
            batch_results = self._batch_llm_validate(batch)

            for candidate, result in zip(batch, batch_results):
                if result is None:
                    continue
                if candidate.ref_index in matched_refs or candidate.target_index in matched_targets:
                    continue

                is_match, confidence, match_type, explanation = result

                if is_match and confidence >= self.strictness:
                    matches[candidate.ref_index] = EntityMatch(
                        reference_entity=candidate.ref_entity,
                        target_entity=candidate.target_entity,
                        confidence=confidence,
                        match_type=match_type,
                        explanation=explanation,
                    )
                    matched_refs.add(candidate.ref_index)
                    matched_targets.add(candidate.target_index)
                    count += 1

        return count

    def _batch_llm_validate(
        self, candidates: List[MatchCandidate]
    ) -> List[Optional[Tuple[bool, float, str, str]]]:
        """
        Validate multiple candidate pairs in a single LLM call.

        Returns:
            List of (is_match, confidence, match_type, explanation) tuples
        """
        if not candidates:
            return []

        # Build batch prompt
        pairs_str = "\n".join([
            f"{i}. REF: \"{c.ref_entity}\" vs TARGET: \"{c.target_entity}\" (vector_sim: {c.similarity:.3f})"
            for i, c in enumerate(candidates)
        ])

        prompt = f"""You are an expert at matching entities semantically.

TASK: Analyze each pair below and determine if they represent the same entity.

Consider variations:
- Accents: "João" = "Joao" = "JOÃO"
- Case: "MARIA SILVA" = "Maria Silva" = "maria silva"
- Order: "Silva, João" = "João Silva"
- Typos: "Apolinario" = "Apolinário"
- Abbreviations: "Dr. João" = "Doutor João"

PAIRS TO ANALYZE:
{pairs_str}

OUTPUT: Return a JSON array with one object per pair:
[
  {{"pair_index": 0, "is_match": true, "confidence": 0.95, "match_type": "semantic", "reason": "Same name, different case"}},
  {{"pair_index": 1, "is_match": false, "confidence": 0.2, "match_type": "no_match", "reason": "Different people"}}
]

Match strictness threshold: {self.strictness}

JSON ARRAY OUTPUT:"""

        try:
            response = self._llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)

            # Extract JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', raw_text)
            if not json_match:
                logger.warning("   ⚠️ Could not parse LLM batch response as JSON array")
                return [None] * len(candidates)

            results_json = json.loads(json_match.group())

            # Map results to candidates
            results = [None] * len(candidates)
            for item in results_json:
                idx = item.get("pair_index", -1)
                if 0 <= idx < len(candidates):
                    results[idx] = (
                        item.get("is_match", False),
                        float(item.get("confidence", 0.0)),
                        item.get("match_type", "no_match"),
                        item.get("reason", ""),
                    )

            return results

        except Exception as e:
            logger.warning(f"   ⚠️ LLM batch validation error: {e}")
            return [None] * len(candidates)

    # Keep the original single-entity methods for backwards compatibility
    def _find_best_match(
        self, ref_entity: str, target_entities: List[str]
    ) -> EntityMatch:
        """Find best matching target entity for reference (legacy method)."""
        # Quick exact match check (case-insensitive)
        ref_normalized = self._normalize_entity(ref_entity)
        for target in target_entities:
            if ref_normalized == self._normalize_entity(target):
                return EntityMatch(
                    reference_entity=ref_entity,
                    target_entity=target,
                    confidence=1.0,
                    match_type="exact",
                    explanation="Exact match (normalized)",
                )

        if not target_entities:
            return EntityMatch(
                reference_entity=ref_entity,
                target_entity="",
                confidence=0.0,
                match_type="no_match",
                explanation="No entities in target document",
            )

        # Use vector similarity for candidates
        top_candidates = self._get_vector_similarity_candidates(ref_entity, target_entities)

        if not top_candidates:
            top_candidates = target_entities[:10]

        prompt = self._build_matching_prompt(ref_entity, top_candidates)
        response = self._llm.invoke(prompt)
        return self._parse_match_response(ref_entity, response, top_candidates)

    def _get_vector_similarity_candidates(
        self, ref_entity: str, target_entities: List[str], top_k: int = 10
    ) -> List[str]:
        """Get top similar candidates using vector similarity."""
        if not self.embeddings:
            return target_entities[:top_k]

        try:
            ref_embedding = self.embeddings.embeddings.embed_query(ref_entity)
            target_embeddings = self.embeddings.embeddings.embed_documents(target_entities)

            ref_vec = np.array(ref_embedding)
            similarities = []

            for i, target_vec in enumerate(target_embeddings):
                target_vec = np.array(target_vec)
                sim = np.dot(ref_vec, target_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(target_vec))
                similarities.append((target_entities[i], sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return [entity for entity, score in similarities[:top_k]]

        except Exception:
            return target_entities[:top_k]

    def _build_matching_prompt(self, ref_entity: str, candidates: List[str]) -> str:
        """Build prompt for semantic matching."""
        candidates_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        return f"""You are an expert at matching entities semantically, handling variations like:
- Accents: "João" matches "Joao"
- Case: "MARIA SILVA" matches "Maria Silva"
- Formatting: "Silva, João" matches "João Silva"
- Typos: "Apolinario" matches "Apolinário"

NOTE: The candidate entities below have been PRE-FILTERED using FAISS + BGE-M3 vector similarity,
so they are already the top-{len(candidates)} most similar matches by semantic embeddings.
Your task is to perform the final semantic validation.

REFERENCE ENTITY: "{ref_entity}"

TOP CANDIDATE ENTITIES (pre-filtered by BGE-M3 embeddings):
{candidates_str}

TASK: Find the best match for the reference entity from these pre-filtered candidates.

OUTPUT FORMAT (strict JSON):
{{
  "match_index": <number or -1 for no match>,
  "confidence": <0.0 to 1.0>,
  "match_type": "<exact|semantic|partial|no_match>",
  "explanation": "<brief reasoning>"
}}

Match strictness threshold: {self.strictness}

JSON OUTPUT:"""

    def _parse_match_response(
        self, ref_entity: str, response, candidates: List[str]
    ) -> EntityMatch:
        """Parse LLM matching response."""
        raw_text = response.content if hasattr(response, "content") else str(response)

        # Extract JSON from response (improved regex for nested JSON)
        json_match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if not json_match:
            return EntityMatch(
                reference_entity=ref_entity,
                target_entity="",
                confidence=0.0,
                match_type="no_match",
                explanation="Failed to parse LLM response",
            )

        try:
            result = json.loads(json_match.group())
            match_idx = result.get("match_index", -1)
            confidence = float(result.get("confidence", 0.0))
            match_type = result.get("match_type", "no_match")
            explanation = result.get("explanation", "")

            if confidence < self.strictness:
                match_type = "no_match"
                target_entity = ""
            elif 0 <= match_idx < len(candidates):
                target_entity = candidates[match_idx]
            else:
                target_entity = ""
                match_type = "no_match"

            return EntityMatch(
                reference_entity=ref_entity,
                target_entity=target_entity,
                confidence=confidence,
                match_type=match_type,
                explanation=explanation,
            )
        except (json.JSONDecodeError, ValueError):
            return EntityMatch(
                reference_entity=ref_entity,
                target_entity="",
                confidence=0.0,
                match_type="no_match",
                explanation="JSON parse error",
            )

    def _create_matching_llm(self, provider: str):
        """Create LLM for semantic matching."""
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=1000,  # Increased for batch responses
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.0,
                max_tokens=1000,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class VerificationEngine:
    """Orchestrates verification workflow."""

    def __init__(self):
        """Initialize verification engine."""
        self.session_manager = VerificationSessionManager()
        self.extractor: Optional[EntityExtractor] = None
        self.matcher: Optional[SemanticMatcher] = None

    def extract_reference(
        self,
        base_document_path: str,
        extraction_query: str,
        llm_provider: str = "openai",
        session_ttl: int = 3600,
    ) -> ReferenceData:
        """
        Extract reference entities from base document.

        Args:
            base_document_path: Path to base reference document
            extraction_query: Query describing what to extract
            llm_provider: LLM provider to use
            session_ttl: Session time-to-live in seconds

        Returns:
            ReferenceData with session information
        """
        # Initialize extractor
        rag_chain = self._create_temp_rag_chain(llm_provider)
        self.extractor = EntityExtractor(rag_chain, llm_provider)

        # Extract entities
        entities, source_chunks = self.extractor.extract(
            document_path=base_document_path, extraction_query=extraction_query
        )

        # Infer entity type from query
        entity_type = self._infer_entity_type(extraction_query)

        # Create session
        session_id = self.session_manager.create_session(
            entity_type=entity_type,
            extraction_query=extraction_query,
            entities=entities,
            base_document=Path(base_document_path).name,
            source_chunks=source_chunks,
            ttl=session_ttl,
        )

        return self.session_manager.get_session(session_id)

    def compare_targets(
        self,
        session_id: str,
        target_document_paths: List[str],
        llm_provider: str = "openai",
        strictness: float = 0.7,
        target_extraction_query: Optional[str] = None,
    ) -> List[VerificationResult]:
        """
        Compare target documents against reference session.

        Args:
            session_id: Session ID from extract_reference
            target_document_paths: List of target document paths
            llm_provider: LLM provider to use
            strictness: Match strictness threshold (0.5-1.0)
            target_extraction_query: Optional custom query for target extraction.
                                    If None, uses the reference extraction query.

        Returns:
            List of VerificationResult, one per target

        Raises:
            ValueError: If session not found or expired
        """
        # Load reference session
        reference = self.session_manager.get_session(session_id)
        if not reference:
            raise ValueError(f"Session {session_id} not found or expired")

        # Initialize embeddings for vector similarity matching
        embeddings_manager = EmbeddingsManager.from_config("config.toml")

        # Initialize matcher with embeddings for FAISS + BGE-M3 support
        self.matcher = SemanticMatcher(llm_provider, strictness, embeddings=embeddings_manager)

        # Determine which query to use for target extraction
        extraction_query_for_target = target_extraction_query if target_extraction_query else reference.extraction_query

        if target_extraction_query:
            logger.info(f"🎯 Using custom target extraction query: '{target_extraction_query}'")
        else:
            logger.info(f"🔄 Using reference extraction query: '{reference.extraction_query}'")

        results = []

        for target_path in target_document_paths:
            result = self._compare_single_target(
                reference=reference,
                target_path=target_path,
                extraction_query=extraction_query_for_target
            )
            results.append(result)

        return results

    def _compare_single_target(
        self, reference: ReferenceData, target_path: str, extraction_query: str
    ) -> VerificationResult:
        """Compare single target document against reference."""
        # Extract entities from target
        rag_chain = self._create_temp_rag_chain(self.matcher.llm_provider)
        target_extractor = EntityExtractor(rag_chain, self.matcher.llm_provider)

        logger.info(f"📄 Extracting entities from target: {Path(target_path).name}")
        target_entities, _ = target_extractor.extract(
            document_path=target_path,
            extraction_query=extraction_query,
            context_name=f"verification_target_{uuid.uuid4().hex[:8]}",
        )
        logger.info(f"✅ Extracted {len(target_entities)} entities from target")

        # Perform semantic matching
        matches, extra_entities = self.matcher.match(
            reference_entities=reference.entities, target_entities=target_entities
        )

        # Analyze results
        missing_entities = [
            m.reference_entity for m in matches if m.match_type == "no_match"
        ]

        matched_count = sum(1 for m in matches if m.match_type != "no_match")
        total_reference = len(reference.entities)

        # Calculate overall confidence
        if total_reference > 0:
            overall_confidence = matched_count / total_reference
        else:
            overall_confidence = 0.0

        # Determine status
        if matched_count == total_reference and not extra_entities:
            status = "verified"
        elif matched_count > 0:
            status = "partial_match"
        else:
            status = "mismatch"

        # Generate summary
        summary = self._generate_summary(
            status=status,
            matched=matched_count,
            total=total_reference,
            missing=len(missing_entities),
            extra=len(extra_entities),
        )

        return VerificationResult(
            target_document=Path(target_path).name,
            status=status,
            matched_entities=matches,
            missing_in_target=missing_entities,
            extra_in_target=extra_entities,
            overall_confidence=overall_confidence,
            summary=summary,
            processed_at=datetime.now(),
            extracted_target_entities=target_entities,  # Store all extracted entities
        )

    def _generate_summary(
        self, status: str, matched: int, total: int, missing: int, extra: int
    ) -> str:
        """Generate human-readable summary."""
        if status == "verified":
            return f"✅ Full verification: {matched}/{total} entities matched perfectly."
        elif status == "partial_match":
            msg = f"⚠️ Partial match: {matched}/{total} entities matched"
            if missing > 0:
                msg += f", {missing} missing"
            if extra > 0:
                msg += f", {extra} extra"
            return msg + "."
        else:
            return f"❌ Mismatch: No entities matched ({total} expected)."

    def _infer_entity_type(self, query: str) -> str:
        """Infer entity type from extraction query."""
        query_lower = query.lower()

        if "name" in query_lower or "employee" in query_lower:
            return "names"
        elif "invoice" in query_lower or "number" in query_lower:
            return "invoice_numbers"
        elif "date" in query_lower:
            return "dates"
        else:
            return "entities"

    def _create_temp_rag_chain(self, llm_provider: str) -> RAGChain:
        """Create temporary RAG chain for verification."""
        embeddings = EmbeddingsManager.from_config("config.toml")
        vector_store = VectorStore(
            embeddings=embeddings.embeddings, context_name="verification_temp"
        )

        return RAGChain.from_config(
            vector_store=vector_store,
            config_path="config.toml",
            llm_provider=llm_provider,
            context_name="verification_temp",
        )
