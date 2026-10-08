"""Choose the 63 face landmarks kept by extract_keypoints.py.

Run once; it prints the list to paste into config.yaml (keypoints.face_indices).
Kept as a script so the choice is reproducible and can be explained.

How it works: MediaPipe describes each face region (lips, eyes, ...) as a set
of edges between landmark indices. We walk along those edges to get the
points in contour order, then take evenly spaced points from that order. This
spreads the points around each shape instead of bunching them up, which
picking by index number would not guarantee.

Usage (from the project root):
    python src/choose_face_indices.py
"""

import numpy as np
from mediapipe.python.solutions import face_mesh_connections

# How many points to keep from each region. MediaPipe has 116 points in these
# regions in total; these counts are roughly proportional to region size
# (63 / 116 = about 54%) and add up to 63.
POINTS_PER_REGION = {
    "FACEMESH_LIPS": 22,  # of 40 (outer + inner lip loops)
    "FACEMESH_LEFT_EYEBROW": 5,  # of 10
    "FACEMESH_RIGHT_EYEBROW": 5,  # of 10
    "FACEMESH_LEFT_EYE": 9,  # of 16
    "FACEMESH_RIGHT_EYE": 9,  # of 16
    "FACEMESH_NOSE": 13,  # of 24
}


def build_neighbours(edges: frozenset) -> dict[int, list[int]]:
    neighbours: dict[int, list[int]] = {}
    for a, b in edges:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    for index in neighbours:
        neighbours[index].sort()
    return neighbours


def walk_in_contour_order(edges: frozenset) -> list[int]:
    """Return all indices of a region, ordered by following its edges.

    A region can have several separate pieces (e.g. outer and inner lip),
    so we walk each piece in turn. A line starts at one of its ends;
    a closed loop starts at its smallest index.
    """
    neighbours = build_neighbours(edges)
    visited: set[int] = set()
    order: list[int] = []

    while len(visited) < len(neighbours):
        unvisited = sorted(i for i in neighbours if i not in visited)
        line_ends = [i for i in unvisited if len(neighbours[i]) == 1]
        start = line_ends[0] if line_ends else unvisited[0]

        # Depth-first walk: keep moving to an unvisited neighbour.
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            order.append(current)
            # Push in reverse so the smallest neighbour is visited first.
            for neighbour in reversed(neighbours[current]):
                if neighbour not in visited:
                    stack.append(neighbour)
    return order


def pick_evenly(ordered: list[int], count: int) -> list[int]:
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[p] for p in positions]


def choose_face_indices() -> list[int]:
    chosen: list[int] = []
    for region_name, count in POINTS_PER_REGION.items():
        edges = getattr(face_mesh_connections, region_name)
        ordered = walk_in_contour_order(edges)
        chosen.extend(pick_evenly(ordered, count))
    return chosen


def main() -> None:
    indices = choose_face_indices()
    # Sanity checks: 63 points and no index chosen twice.
    assert len(indices) == 63, len(indices)
    assert len(set(indices)) == 63, "duplicate face index chosen"
    print(f"# {len(indices)} face indices, in region order: " + ", ".join(POINTS_PER_REGION))
    print(f"face_indices: {indices}")


if __name__ == "__main__":
    main()
