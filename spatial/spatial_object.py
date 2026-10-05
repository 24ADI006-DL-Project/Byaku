from typing import TypedDict


class SpatialObject(TypedDict):
    """Tracked-object data enriched with relative depth when available."""

    track_id: int
    class_name: str
    confidence: float
    bounding_box: list[float]
    relative_depth: float | None