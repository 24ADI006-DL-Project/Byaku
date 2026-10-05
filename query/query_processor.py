from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict


QueryIntent = Literal["locate", "existence"]
QueryStatus = Literal["ok", "empty_query", "unsupported", "ambiguous", "unknown_object"]


class ParsedQuery(TypedDict):
    intent: QueryIntent | None
    target: str | None
    status: QueryStatus


DEFAULT_OBJECT_VOCABULARY: dict[str, tuple[str, ...]] = {
    "person": ("person", "people"),
    "bottle": ("bottle",),
    "chair": ("chair",),
    "laptop": ("laptop", "notebook computer"),
    "table": ("table",),
    "door": ("door",),
    "phone": ("phone", "cell phone", "mobile phone"),
    "cup": ("cup",),
    "book": ("book",),
    "bag": ("bag", "backpack"),
    "car": ("car", "vehicle"),
    "stairs": ("stairs", "staircase"),
    "sofa": ("sofa", "couch"),
    "monitor": ("monitor", "screen"),
}

_STOP_WORDS = {
    "a", "an", "are", "do", "does", "find", "is", "locate", "me", "my",
    "please", "see", "show", "the", "there", "where", "you", "any", "in",
    "front", "of", "scene", "right", "now", "near", "can", "we", "have",
}


class QueryProcessor:
    """Parse simple locate/existence questions using deterministic word matching."""

    def __init__(
        self,
        object_vocabulary: Iterable[str] | Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._forms: dict[str, set[str]] = {}
        vocabulary = DEFAULT_OBJECT_VOCABULARY if object_vocabulary is None else object_vocabulary

        if isinstance(vocabulary, Mapping):
            entries = vocabulary.items()
        else:
            if isinstance(vocabulary, str):
                vocabulary = [vocabulary]
            entries = ((name, ()) for name in vocabulary)

        for canonical_name, aliases in entries:
            canonical = self._normalize(str(canonical_name))
            if not canonical:
                continue
            forms = self._forms.setdefault(canonical, set())
            values = [canonical_name]
            if isinstance(aliases, str):
                values.append(aliases)
            else:
                values.extend(aliases)
            for value in values:
                normalized = self._normalize(str(value))
                if normalized:
                    forms.add(normalized)
                    plural = self._plural_form(normalized)
                    if plural:
                        forms.add(plural)

    def parse(self, query: Any) -> ParsedQuery:
        if not isinstance(query, str) or not query.strip():
            return {"intent": None, "target": None, "status": "empty_query"}

        normalized = self._normalize(query)
        intent = self._intent_for(normalized)
        if intent is None:
            return {"intent": None, "target": None, "status": "unsupported"}

        matches = self._vocabulary_matches(normalized)
        if len(matches) == 1:
            return {"intent": intent, "target": matches[0], "status": "ok"}
        if len(matches) > 1:
            return {"intent": intent, "target": None, "status": "ambiguous"}

        target = self._unknown_target(normalized)
        return {"intent": intent, "target": target, "status": "unknown_object"}

    def canonicalize_class_name(self, class_name: Any) -> str:
        normalized = self._normalize(str(class_name))
        for canonical, forms in self._forms.items():
            if normalized in forms:
                return canonical
        return normalized

    def _vocabulary_matches(self, query: str) -> list[str]:
        padded_query = f" {query} "
        matches = {
            canonical
            for canonical, forms in self._forms.items()
            if any(f" {form} " in padded_query for form in forms)
        }
        return sorted(matches)

    @staticmethod
    def _intent_for(query: str) -> QueryIntent | None:
        if re.search(r"\b(where|locate|find)\b", query):
            return "locate"
        if re.search(r"\b(is there|are there|do you see|can you see|any)\b", query):
            return "existence"
        return None

    @staticmethod
    def _unknown_target(query: str) -> str | None:
        words = [word for word in query.split() if word not in _STOP_WORDS]
        return " ".join(words) or None

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    @classmethod
    def _plural_form(cls, phrase: str) -> str:
        words = phrase.split()
        if not words:
            return ""
        last = words[-1]
        if last.endswith(("s", "x", "z", "ch", "sh")):
            words[-1] = f"{last}es"
        elif len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
            words[-1] = f"{last[:-1]}ies"
        else:
            words[-1] = f"{last}s"
        return " ".join(words)