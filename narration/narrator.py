from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Narrator:
    """Turn Module 8 query results into concise deterministic speech text."""

    _position_phrases = {
        "left": "on your left",
        "center": "in front of you",
        "right": "on your right",
    }
    _distance_phrases = {
        "very_close": "very close",
        "close": "close",
        "moderate": "moderately far",
        "far": "far",
    }
    _irregular_plurals = {"person": "people", "child": "children", "mouse": "mice"}

    def narrate(self, query_result: Any) -> str:
        """Create a short spoken response from a structured spatial query result."""
        if not isinstance(query_result, Mapping):
            return "I can't determine the scene information right now."

        status = query_result.get("status")
        if status == "empty_query":
            return "Please ask about an object."
        if status == "unsupported":
            return "I can only locate an object or check whether it is visible."
        if status == "ambiguous":
            return "Please ask about one object at a time."

        objects_value = query_result.get("objects")
        objects = [obj for obj in objects_value if isinstance(obj, Mapping)] if isinstance(
            objects_value, (list, tuple)
        ) else []
        if "found" not in query_result or not isinstance(query_result.get("found"), bool):
            return "I can't determine the scene information right now."
        if not query_result["found"]:
            return self._not_found(query_result.get("target"))
        if not objects:
            return "I can't determine the object's location from the available information."

        names = [self._object_name(obj, query_result.get("target")) for obj in objects]
        if len(objects) == 1:
            return self._narrate_one(
                names[0],
                objects[0],
                query_result.get("intent") == "existence",
            )

        return self._narrate_many(
            names[0],
            objects,
            query_result.get("intent") == "existence",
        )

    def _narrate_one(self, name: str, obj: Mapping[str, Any], existence: bool) -> str:
        location = self._location(obj)
        if not location:
            if existence:
                return f"Yes, I can see a {name}, but its location is unclear."
            return f"I can see your {name}, but its location is unclear."

        if existence:
            article = "an" if name[:1].lower() in "aeiou" else "a"
            return f"Yes, there is {article} {name} {location}."
        return f"Your {name} is {location}."

    def _narrate_many(self, name: str, objects: list[Mapping[str, Any]], existence: bool) -> str:
        count = len(objects)
        plural_name = self._pluralize(name)
        intro = f"There are {self._number_word(count)} {plural_name}."
        if existence:
            intro = f"Yes, {intro[0].lower()}{intro[1:]}"

        descriptions = [self._location(obj, multiple=True) or "in an unclear location" for obj in objects]
        details = [f"One is {descriptions[0]}"]
        details.extend(f"one is {description}" for description in descriptions[1:])
        return f"{intro} " + " and ".join(details) + "."

    def _location(self, obj: Mapping[str, Any], multiple: bool = False) -> str:
        distance = self._distance_phrases.get(obj.get("distance_category"))
        position = self._position_phrases.get(obj.get("horizontal_position"))

        if multiple and distance == "far":
            distance = "farther"
        if distance and position:
            return f"{distance} {position}" if multiple else f"{distance} and {position}"
        return distance or position or ""

    @staticmethod
    def _object_name(obj: Mapping[str, Any], target: Any) -> str:
        value = obj.get("class_name") or target
        if not isinstance(value, str) or not value.strip():
            return "object"
        return " ".join(value.strip().replace("_", " ").split())

    @staticmethod
    def _not_found(target: Any) -> str:
        if isinstance(target, str) and target.strip():
            name = " ".join(target.strip().replace("_", " ").split())
            return f"The {name} is not currently visible."
        return "The requested object is not currently visible."

    @classmethod
    def _pluralize(cls, name: str) -> str:
        if name in cls._irregular_plurals:
            return cls._irregular_plurals[name]
        words = name.split()
        last = words[-1]
        if last.endswith(("s", "x", "z", "ch", "sh")):
            words[-1] = f"{last}es"
        elif len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
            words[-1] = f"{last[:-1]}ies"
        else:
            words[-1] = f"{last}s"
        return " ".join(words)

    @staticmethod
    def _number_word(value: int) -> str:
        words = {2: "two", 3: "three", 4: "four", 5: "five"}
        return words.get(value, str(value))