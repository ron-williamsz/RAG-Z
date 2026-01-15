"""Session manager for document verification with TTL-based expiration and disk persistence."""

import uuid
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from threading import Lock

from .verification_models import ReferenceData


class VerificationSessionManager:
    """Manages verification sessions with TTL and disk persistence."""

    def __init__(self, persist_dir: str = "data/verification_sessions"):
        """Initialize session manager with disk persistence."""
        self._sessions: Dict[str, ReferenceData] = {}
        self._lock = Lock()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Load existing sessions from disk on startup
        self._load_sessions_from_disk()

    def create_session(
        self,
        entity_type: str,
        extraction_query: str,
        entities: List[str],
        base_document: str,
        source_chunks: List[dict],
        ttl: int = 3600,
    ) -> str:
        """
        Create new verification session.

        Args:
            entity_type: Type of entities (e.g., "names", "invoice_numbers")
            extraction_query: Original extraction query
            entities: List of extracted entities
            base_document: Name of base reference document
            source_chunks: Source chunks where entities were found
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid.uuid4())

        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl)

        reference_data = ReferenceData(
            session_id=session_id,
            entity_type=entity_type,
            extraction_query=extraction_query,
            entities=entities,
            base_document=base_document,
            source_chunks=source_chunks,
            created_at=now,
            expires_at=expires_at,
        )

        with self._lock:
            self._sessions[session_id] = reference_data
            self._cleanup_expired()
            # Persist to disk
            self._save_session_to_disk(session_id, reference_data)

        return session_id

    def get_session(self, session_id: str) -> Optional[ReferenceData]:
        """
        Retrieve session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ReferenceData if found and not expired, None otherwise
        """
        with self._lock:
            self._cleanup_expired()
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session manually.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                # Delete from disk
                self._delete_session_from_disk(session_id)
                return True
            return False

    def _cleanup_expired(self):
        """Remove expired sessions (internal, not thread-safe by itself)."""
        now = datetime.now()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.expires_at < now
        ]
        for sid in expired:
            del self._sessions[sid]
            # Delete from disk
            self._delete_session_from_disk(sid)

    def get_active_sessions(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active (non-expired) sessions
        """
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)

    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        with self._lock:
            self._cleanup_expired()
            return list(self._sessions.keys())

    def _save_session_to_disk(self, session_id: str, session_data: ReferenceData):
        """Save session to disk as JSON."""
        try:
            session_file = self.persist_dir / f"{session_id}.json"
            session_dict = {
                "session_id": session_data.session_id,
                "entity_type": session_data.entity_type,
                "extraction_query": session_data.extraction_query,
                "entities": session_data.entities,
                "base_document": session_data.base_document,
                "source_chunks": session_data.source_chunks,
                "created_at": session_data.created_at.isoformat(),
                "expires_at": session_data.expires_at.isoformat(),
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_dict, f, ensure_ascii=False, indent=2)
        except Exception:
            # Non-critical: if disk save fails, session still exists in memory
            pass

    def _load_sessions_from_disk(self):
        """Load all sessions from disk on startup."""
        try:
            for session_file in self.persist_dir.glob("*.json"):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        session_dict = json.load(f)

                    # Parse datetime strings
                    created_at = datetime.fromisoformat(session_dict["created_at"])
                    expires_at = datetime.fromisoformat(session_dict["expires_at"])

                    # Skip if already expired
                    if expires_at < datetime.now():
                        session_file.unlink()  # Delete expired file
                        continue

                    # Reconstruct ReferenceData
                    session_data = ReferenceData(
                        session_id=session_dict["session_id"],
                        entity_type=session_dict["entity_type"],
                        extraction_query=session_dict["extraction_query"],
                        entities=session_dict["entities"],
                        base_document=session_dict["base_document"],
                        source_chunks=session_dict["source_chunks"],
                        created_at=created_at,
                        expires_at=expires_at,
                    )

                    self._sessions[session_data.session_id] = session_data
                except Exception:
                    # Skip corrupted session files
                    continue
        except Exception:
            # If directory doesn't exist or other error, just start fresh
            pass

    def _delete_session_from_disk(self, session_id: str):
        """Delete session file from disk."""
        try:
            session_file = self.persist_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
        except Exception:
            # Non-critical
            pass
