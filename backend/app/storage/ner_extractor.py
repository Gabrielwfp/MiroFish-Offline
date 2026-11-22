"""
NER/RE Extractor — entity and relation extraction via local LLM

Replaces Zep Cloud's built-in NER/RE pipeline.
Uses LLMClient.chat_json() with a JSON Schema built from the graph's ontology
(enforced by Ollama's structured outputs) and validates every response against
the NERResult contract. Failures raise NERExtractionError instead of silently
returning an empty extraction.
"""

import logging
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..utils.llm_client import ContextOverflowError, LLMClient

logger = logging.getLogger('mirofish.ner_extractor')


class NERExtractionError(RuntimeError):
    """Raised when the LLM cannot produce a contract-compliant extraction."""


# --- Response contract -------------------------------------------------------

class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: str = Field(min_length=1)
    fact: str = ""


class NERResult(BaseModel):
    # Unknown top-level keys (e.g. "relationships" instead of "relations") are a contract violation
    model_config = ConfigDict(extra="forbid")
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


def _type_names(items: List[Any]) -> List[str]:
    """Names of ontology entity/relation types (items may be dicts or plain strings)."""
    names = []
    for it in items or []:
        name = (it.get("name", "") if isinstance(it, dict) else str(it)).strip()
        if name and name not in names:
            names.append(name)
    return names

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """You are a Named Entity Recognition and Relation Extraction system.
Given a text and an ontology (entity types + relation types), extract all entities and relations.

ONTOLOGY:
{ontology_description}

RULES:
1. Only extract entity types and relation types defined in the ontology.
2. Normalize entity names: strip whitespace, use canonical form (e.g., "Jack Ma" not "ma jack").
3. Each entity must have: name, type (from ontology), and optional attributes.
4. Each relation must have: source entity name, target entity name, type (from ontology), and a fact sentence describing the relationship.
5. If no entities or relations are found, return empty lists.
6. Be precise — only extract what is explicitly stated or strongly implied in the text.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """Extract entities and relations from the following text:

{text}"""


class NERExtractor:
    """Extract entities and relations from text using local LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relations from text, guided by ontology.

        Args:
            text: Input text chunk
            ontology: Dict with 'entity_types' and 'relation_types' from graph

        Returns:
            Dict with 'entities' and 'relations' lists:
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
            Empty lists only when the model validly found nothing.

        Raises:
            NERExtractionError: if no contract-compliant response is obtained
                within max_retries + 1 attempts.
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        schema = self._build_schema(ontology)
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                raw = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,  # Low temp for extraction precision
                    max_tokens=4096,
                    schema=schema,
                    disable_thinking=True,
                    operation="ner",
                )
                result = NERResult.model_validate(raw)
                return self._validate_and_clean(result.model_dump(), ontology)

            except ContextOverflowError as e:  # retrying cannot help; fail the chunk visibly
                raise NERExtractionError(f"NER chunk does not fit the context window: {e}") from e
            except ValidationError as e:  # must precede ValueError (it subclasses it)
                last_error = e
                logger.warning(
                    f"NER response violates contract (attempt {attempt}/{attempts}): {str(e)[:500]}"
                )
            except ValueError as e:
                last_error = e
                logger.warning(f"NER returned invalid JSON (attempt {attempt}/{attempts}): {str(e)[:500]}")
            except Exception as e:
                last_error = e
                logger.error(f"NER LLM call failed (attempt {attempt}/{attempts}): {e}")

        raise NERExtractionError(
            f"NER extraction failed after {attempts} attempts "
            f"(model={self.llm.model}): {str(last_error)[:500]}"
        )

    def _build_schema(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """JSON Schema for the NERResult contract; ontology types become enums."""
        entity_types = _type_names(ontology.get("entity_types", []))
        relation_types = _type_names(ontology.get("relation_types", ontology.get("edge_types", [])))

        def type_field(names: List[str]) -> Dict[str, Any]:
            return {"type": "string", "enum": names} if names else {"type": "string"}

        return {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": type_field(entity_types),
                            "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                        },
                        "required": ["name", "type", "attributes"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": type_field(relation_types),
                            "fact": {"type": "string"},
                        },
                        "required": ["source", "target", "type", "fact"],
                    },
                },
            },
            "required": ["entities", "relations"],
        }

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Format ontology dict into readable text for the LLM prompt."""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("Entity Types:")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (attributes: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\nRelation Types:")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if source_targets:
                        st_strs = [f"{st.get('source', '?')} → {st.get('target', '?')}" for st in source_targets]
                        line += f" ({', '.join(st_strs)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("No specific ontology defined. Extract all entities and relations you find.")

        return "\n".join(parts)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and normalize LLM output."""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Get valid type names from ontology
        valid_entity_types = set(_type_names(ontology.get("entity_types", [])))

        # Clean entities
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entity")).strip()
            if not name:
                continue

            # Deduplicate by normalized name
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # If ontology has types, warn but keep entities with unknown types
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"Entity '{name}' has type '{etype}' not in ontology, keeping anyway")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # Clean relations
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # Ensure source and target entities exist
            # (they might not if LLM hallucinated a relation without the entity)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
