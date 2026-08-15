# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 ToPu
#
# The transfer workflow was independently reimplemented after studying
# Blender-CM3D2-Converter, licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.

"""Transfer relative or absolute shape keys between different mesh topologies."""

from __future__ import annotations

from fnmatch import fnmatchcase
import time
import traceback

import bpy
import numpy as np
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Object, Operator, Panel, PropertyGroup
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


# Extensions read blender_manifest.toml and ignore this, but a copy dropped into
# scripts/addons is only recognised as an add-on when bl_info is present.
bl_info = {
    "name": "Forced Shape Key Transfer",
    "author": "Boon56228",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Object Data Properties > Forced Shape Key Transfer",
    "description": "Transfer shape keys between meshes with different topology",
    "category": "Object",
}


ADDON_ID = "topu_forced_shape_key_transfer"

# Target vertices handled between two generator yields while mapping is built.
_MAPPING_CHUNK = 2048

# Nearest-surface mapping needs faces. A source without any falls back to
# nearest-vertex sampling, blended over this many neighbours.
_FALLBACK_BLEND_COUNT = 3

# Mapping results are keyed by geometry content, so stale entries can never be
# reused after an edit. Only a handful of pairs are worth keeping resident.
_MAPPING_CACHE: dict[tuple, tuple] = {}
_MAPPING_CACHE_LIMIT = 4

_LAST_ANALYSIS: list[str] = []


def _tr(text: str) -> str:
    return bpy.app.translations.pgettext_iface(text)


def _settings(context):
    """None while the add-on is not fully registered.

    A stale panel left behind by a failed reload still gets drawn, so every
    entry point has to cope with the scene property being absent rather than
    raising on each redraw.
    """
    scene = getattr(context, "scene", None)
    return getattr(scene, "topu_fst_settings", None) if scene is not None else None


# ---------------------------------------------------------------------------
# Selection resolution
# ---------------------------------------------------------------------------


def _resolve_source_targets(context, settings):
    """Return (source, targets, message). Supports one source to many targets."""
    active = context.active_object
    if active is not None and active.mode != "OBJECT":
        return None, [], "Switch to Object Mode before transferring."

    selected = [obj for obj in context.selected_objects if obj.type == "MESH"]
    override = settings.source_override if settings is not None else None

    if override is not None and override.type == "MESH":
        source = override
        targets = [obj for obj in selected if obj != source]
        if active is not None and active.type == "MESH" and active != source and active not in targets:
            targets.append(active)
        if not targets:
            return None, [], "Select at least one target mesh besides the source."
    else:
        if active is None or active.type != "MESH":
            return None, [], "The active object must be a mesh."
        if len(selected) != 2:
            return None, [], "Select exactly two mesh objects, or pick an explicit source."
        candidates = [obj for obj in selected if obj != active]
        if len(candidates) != 1:
            return None, [], "The source and target must both be mesh objects."
        source = candidates[0]
        targets = [active]

    source_keys = source.data.shape_keys
    if source_keys is None or len(source_keys.key_blocks) < 2:
        return None, [], "The source has no transferable shape keys."
    if len(source.data.vertices) == 0:
        return None, [], "The source and target meshes must contain vertices."
    if any(len(obj.data.vertices) == 0 for obj in targets):
        return None, [], "The source and target meshes must contain vertices."

    # Preserve selection order for reproducible reports.
    ordered = []
    for obj in targets:
        if obj not in ordered:
            ordered.append(obj)
    return source, ordered, ""


# ---------------------------------------------------------------------------
# NumPy helpers
# ---------------------------------------------------------------------------


def _vector_array(collection, count: int, attribute: str = "co") -> np.ndarray:
    flat = np.empty(count * 3, dtype=np.float32)
    collection.foreach_get(attribute, flat)
    return flat.reshape(count, 3).astype(np.float64)


def _mesh_basis_coords(mesh, ignore_shape_keys: bool = False) -> np.ndarray:
    count = len(mesh.vertices)
    if not ignore_shape_keys and mesh.shape_keys and mesh.shape_keys.key_blocks:
        return _vector_array(mesh.shape_keys.key_blocks[0].data, count)
    return _vector_array(mesh.vertices, count)


def _transform_points(points: np.ndarray, matrix) -> np.ndarray:
    matrix_array = np.array(matrix, dtype=np.float64)
    return points @ matrix_array[:3, :3].T + matrix_array[:3, 3]


def _matrix3(matrix) -> np.ndarray:
    return np.array(matrix, dtype=np.float64)[:3, :3]


def _vertex_normals(mesh, count: int) -> np.ndarray | None:
    try:
        return _vector_array(mesh.vertex_normals, count, "vector")
    except (AttributeError, RuntimeError, TypeError):
        return None


def _vertex_group_weights(obj, group_name: str) -> np.ndarray | None:
    group = obj.vertex_groups.get(group_name) if group_name else None
    if group is None:
        return None
    group_index = group.index
    weights = np.zeros(len(obj.data.vertices), dtype=np.float64)
    for vertex in obj.data.vertices:
        for element in vertex.groups:
            if element.group == group_index:
                weights[vertex.index] = element.weight
                break
    return weights


def _evaluated_coords(obj, depsgraph, expected_vertices: int, expected_polygons: int):
    """Evaluated coordinates, but only when modifiers preserved the index layout."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = None
    try:
        mesh = evaluated.to_mesh()
        if mesh is None:
            return None
        if len(mesh.vertices) != expected_vertices or len(mesh.polygons) != expected_polygons:
            return None
        return _vector_array(mesh.vertices, expected_vertices)
    except RuntimeError:
        return None
    finally:
        if mesh is not None:
            evaluated.to_mesh_clear()


# ---------------------------------------------------------------------------
# Source snapshot
# ---------------------------------------------------------------------------


def _snapshot_source(source):
    mesh = source.data
    shape_keys = mesh.shape_keys
    vertex_count = len(mesh.vertices)
    records = []

    for key_block in shape_keys.key_blocks:
        relative = key_block.relative_key
        records.append(
            {
                "name": key_block.name,
                "relative_name": relative.name if relative else shape_keys.key_blocks[0].name,
                "coords": _vector_array(key_block.data, vertex_count),
                "slider_min": key_block.slider_min,
                "slider_max": key_block.slider_max,
                "value": key_block.value,
                "mute": key_block.mute,
                "interpolation": key_block.interpolation,
                "vertex_group": key_block.vertex_group,
                "frame": key_block.frame,
            }
        )

    return {
        "records": records,
        "use_relative": shape_keys.use_relative,
        "eval_time": shape_keys.eval_time,
        "vertex_count": vertex_count,
    }


def _key_selected(name: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return True
    for token in pattern.split(","):
        token = token.strip()
        if token and fnmatchcase(name, token):
            return True
    return False


# ---------------------------------------------------------------------------
# Triangle frames
# ---------------------------------------------------------------------------


def _triangle_frames(corner_a: np.ndarray, corner_b: np.ndarray, corner_c: np.ndarray) -> np.ndarray:
    """Orthonormal column-major local frames, one per triangle.

    Rebuilding a target vertex's offset in this frame makes the offset follow
    the surface's rotation while keeping its length, so garment thickness turns
    with the body instead of only sliding with it. Because the frame is
    orthonormal its inverse is just its transpose, and sliver triangles cannot
    blow it up the way a stretch-carrying frame would.
    """
    edge_0 = corner_b - corner_a
    edge_1 = corner_c - corner_a
    cross = np.cross(edge_0, edge_1)
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1), 1.0e-24)[:, None]
    tangent = edge_0 / np.maximum(np.linalg.norm(edge_0, axis=1), 1.0e-24)[:, None]
    bitangent = np.cross(normal, tangent)
    return np.stack((tangent, bitangent, normal), axis=2)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class _Mapping:
    __slots__ = (
        "kind",
        "triangles",
        "triangle_index",
        "barycentric",
        "local_offset",
        "vertex_index",
        "vertex_weight",
        "weight",
        "valid",
        "distance",
        "missed",
    )

    def __init__(self, kind: str):
        self.kind = kind
        self.triangles = None
        self.triangle_index = None
        self.barycentric = None
        self.local_offset = None
        self.vertex_index = None
        self.vertex_weight = None
        self.weight = None
        self.valid = None
        self.distance = None
        self.missed = 0


def _distance_weights(distance: np.ndarray, max_distance: float | None, falloff: float):
    """Smooth ramp to zero so the distance limit does not tear the surface.

    `max_distance` is None when unlimited. A limit of exactly zero is honoured
    literally: only vertices already lying on the source can match.
    """
    if max_distance is None:
        return np.ones_like(distance), np.ones(distance.shape, dtype=bool)

    valid = distance <= max_distance
    inner = max_distance * (1.0 - falloff)
    weight = np.ones_like(distance)
    if max_distance > inner:
        ramp = np.clip((max_distance - distance) / (max_distance - inner), 0.0, 1.0)
        smooth = ramp * ramp * (3.0 - 2.0 * ramp)
        weight = np.where(distance > inner, smooth, 1.0)
    weight = np.where(valid, weight, 0.0)
    return weight, valid


def _source_triangles(mesh):
    # Blender 4.1 computes loop triangles lazily and deprecated the explicit
    # call, so treat it as best effort.
    try:
        mesh.calc_loop_triangles()
    except (AttributeError, RuntimeError):
        pass
    count = len(mesh.loop_triangles)
    if count == 0:
        return None
    vertices = np.empty(count * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", vertices)
    return vertices.reshape(count, 3)


def _drop_degenerate(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Zero-area triangles have no usable normal, so keep them out of the BVH."""
    corner_a = points[triangles[:, 0]]
    corner_b = points[triangles[:, 1]]
    corner_c = points[triangles[:, 2]]
    cross = np.cross(corner_b - corner_a, corner_c - corner_a)
    area = np.linalg.norm(cross, axis=1)
    scale = float(np.max(np.ptp(points, axis=0))) if len(points) else 1.0
    keep = area > max(scale * scale * 1.0e-12, 1.0e-24)
    return triangles if keep.all() else triangles[keep]


def _compact_triangles(mapping: _Mapping, triangles: np.ndarray, triangle_index: np.ndarray):
    """Keep only the triangles that were actually hit, so per-key work is small."""
    used, remapped = np.unique(triangle_index, return_inverse=True)
    mapping.triangles = triangles[used]
    mapping.triangle_index = remapped.astype(np.int32)


def _finalize_surface_mapping(
    mapping: _Mapping,
    triangles: np.ndarray,
    triangle_index: np.ndarray,
    barycentric: np.ndarray,
    target_points: np.ndarray,
    source_points: np.ndarray,
):
    _compact_triangles(mapping, triangles, triangle_index)
    mapping.barycentric = barycentric

    tris = mapping.triangles
    corner_a = source_points[tris[:, 0]]
    corner_b = source_points[tris[:, 1]]
    corner_c = source_points[tris[:, 2]]
    # Orthonormal frames invert by transposition.
    inverses = np.transpose(_triangle_frames(corner_a, corner_b, corner_c), (0, 2, 1))

    surface = (
        corner_a[mapping.triangle_index] * barycentric[:, 0:1]
        + corner_b[mapping.triangle_index] * barycentric[:, 1:2]
        + corner_c[mapping.triangle_index] * barycentric[:, 2:3]
    )
    offset = target_points - surface
    local = np.einsum("nij,nj->ni", inverses[mapping.triangle_index], offset)
    # Unmapped vertices point at a placeholder triangle; keep their offsets
    # finite so a later multiply by a zero weight cannot produce NaN.
    local[~mapping.valid] = 0.0
    mapping.local_offset = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)


def _barycentric_weights(point, corner_a, corner_b, corner_c):
    """Weights of a point already known to lie on the triangle.

    The hit returned by the BVH sits on the triangle, so the result is clamped
    and renormalised only to absorb floating-point excursions at edges.
    """
    point = np.asarray(point, dtype=np.float64)
    corner_a = np.asarray(corner_a, dtype=np.float64)
    edge_0 = np.asarray(corner_b, dtype=np.float64) - corner_a
    edge_1 = np.asarray(corner_c, dtype=np.float64) - corner_a
    point_vector = point - corner_a
    dot_00 = edge_0.dot(edge_0)
    dot_01 = edge_0.dot(edge_1)
    dot_11 = edge_1.dot(edge_1)
    dot_20 = point_vector.dot(edge_0)
    dot_21 = point_vector.dot(edge_1)
    denominator = dot_00 * dot_11 - dot_01 * dot_01
    if abs(denominator) <= 1.0e-20:
        return 1.0, 0.0, 0.0
    weight_b = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    weight_c = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    weight_a = 1.0 - weight_b - weight_c
    weights = [max(0.0, min(1.0, value)) for value in (weight_a, weight_b, weight_c)]
    total = sum(weights)
    if total <= 1.0e-20:
        return 1.0, 0.0, 0.0
    return weights[0] / total, weights[1] / total, weights[2] / total


def _pick_candidate(candidates, normal, normal_weight: float, epsilon: float):
    """Choose among competing hits by distance and normal agreement.

    Without this a target vertex snaps to whatever surface is nearest, which is
    the wrong side wherever two sheets nearly touch: lips, eyelids, the inside
    of a thin panel.
    """
    usable = [item for item in candidates if item[2] is not None]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    # Scale the penalty by the spread of the candidates so it stays comparable
    # to the distances being traded off.
    reference = max(max(item[3] for item in usable), epsilon)
    best = None
    best_score = None
    for item in usable:
        _point, candidate_normal, _index, distance = item
        alignment = normal[0] * candidate_normal[0] + normal[1] * candidate_normal[1] + normal[2] * candidate_normal[2]
        score = distance + normal_weight * reference * (1.0 - alignment)
        if best_score is None or score < best_score:
            best_score = score
            best = item
    return best


def _build_surface_mapping(
    source_points: np.ndarray,
    triangles: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray | None,
    max_distance: float | None,
    falloff: float,
    normal_weight: float,
):
    """Ray along the target normal first, then fall back to scored nearest hits.

    Both stages weigh normal agreement, so a vertex never snaps to the back of
    a thin panel or across the gap between lips or eyelids just because that
    surface happens to be marginally closer.
    """
    triangles = _drop_degenerate(triangles, source_points)
    if len(triangles) == 0:
        return None

    # tolist() hands mathutils plain Python numbers; NumPy scalars are not
    # accepted as polygon indices.
    tree = BVHTree.FromPolygons(source_points.tolist(), triangles.tolist(), all_triangles=True)

    count = len(target_points)
    triangle_index = np.zeros(count, dtype=np.int64)
    barycentric = np.zeros((count, 3), dtype=np.float64)
    distance = np.full(count, np.inf, dtype=np.float64)
    found = np.zeros(count, dtype=bool)

    scale_hint = float(np.max(np.ptp(source_points, axis=0))) if len(source_points) else 1.0
    epsilon = max(scale_hint * 1.0e-6, 1.0e-9)
    use_normals = target_normals is not None

    for start in range(0, count, _MAPPING_CHUNK):
        stop = min(start + _MAPPING_CHUNK, count)
        for index in range(start, stop):
            position = target_points[index]
            origin = (float(position[0]), float(position[1]), float(position[2]))
            hit_index = None
            hit_point = None
            hit_distance = None

            if use_normals:
                # Both directions are cast, then scored: the nearer hit is often
                # the back face of the sheet the vertex actually belongs to.
                normal = target_normals[index]
                rays = []
                for sign in (-1.0, 1.0):
                    direction = (
                        float(normal[0]) * sign,
                        float(normal[1]) * sign,
                        float(normal[2]) * sign,
                    )
                    ray = tree.ray_cast(origin, direction, max_distance if max_distance is not None else 1.0e18)
                    if ray[0] is not None and ray[2] is not None:
                        rays.append(ray)
                chosen = _pick_candidate(rays, normal, normal_weight, epsilon)
                if chosen is not None:
                    hit_point, _ray_normal, hit_index, hit_distance = chosen

            if hit_index is None:
                nearest = (
                    tree.find_nearest(origin, max_distance)
                    if max_distance is not None
                    else tree.find_nearest(origin)
                )
                if nearest is None or nearest[0] is None or nearest[2] is None:
                    continue
                hit_point, _nearest_normal, hit_index, hit_distance = nearest

                if use_normals:
                    radius = max_distance if max_distance is not None else hit_distance * 2.0 + epsilon
                    chosen = _pick_candidate(
                        tree.find_nearest_range(origin, radius), target_normals[index], normal_weight, epsilon
                    )
                    if chosen is not None:
                        hit_point, _chosen_normal, hit_index, hit_distance = chosen

            corners = triangles[hit_index]
            weights = _barycentric_weights(
                hit_point,
                source_points[corners[0]],
                source_points[corners[1]],
                source_points[corners[2]],
            )
            triangle_index[index] = hit_index
            barycentric[index] = weights
            distance[index] = hit_distance
            found[index] = True
        yield stop / count

    mapping = _Mapping("SURFACE")
    weight, within = _distance_weights(distance, max_distance, falloff)
    mapping.valid = found & within
    mapping.weight = np.where(mapping.valid, weight, 0.0)
    mapping.distance = distance
    mapping.missed = int(count - int(np.count_nonzero(mapping.valid)))

    triangle_index = np.where(mapping.valid, triangle_index, 0)
    barycentric = np.where(mapping.valid[:, None], barycentric, np.array([1.0, 0.0, 0.0]))
    _finalize_surface_mapping(mapping, triangles, triangle_index, barycentric, target_points, source_points)
    return mapping


def _build_vertex_mapping(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float | None,
    falloff: float,
    blend_count: int = _FALLBACK_BLEND_COUNT,
):
    """Fallback for sources without faces: inverse-distance blend of neighbours."""
    tree = KDTree(len(source_points))
    for index, position in enumerate(source_points):
        tree.insert((float(position[0]), float(position[1]), float(position[2])), index)
    tree.balance()

    count = len(target_points)
    neighbours = max(1, min(blend_count, len(source_points)))
    vertex_index = np.zeros((count, neighbours), dtype=np.int32)
    vertex_weight = np.zeros((count, neighbours), dtype=np.float64)
    distance = np.full(count, np.inf, dtype=np.float64)
    found = np.zeros(count, dtype=bool)

    scale_hint = float(np.max(np.ptp(source_points, axis=0))) if len(source_points) else 1.0
    epsilon = max(scale_hint * 1.0e-9, 1.0e-12)

    for start in range(0, count, _MAPPING_CHUNK):
        stop = min(start + _MAPPING_CHUNK, count)
        for index in range(start, stop):
            position = target_points[index]
            origin = (float(position[0]), float(position[1]), float(position[2]))
            if neighbours == 1:
                _position, source_index, nearest_distance = tree.find(origin)
                if source_index is None:
                    continue
                vertex_index[index, 0] = source_index
                vertex_weight[index, 0] = 1.0
                distance[index] = nearest_distance
                found[index] = True
                continue

            hits = tree.find_n(origin, neighbours)
            if not hits:
                continue
            weights = []
            for slot, (_position, source_index, hit_distance) in enumerate(hits):
                vertex_index[index, slot] = source_index
                weights.append(1.0 / (hit_distance + epsilon))
            total = sum(weights)
            for slot, value in enumerate(weights):
                vertex_weight[index, slot] = value / total
            # Pad short results by repeating the nearest hit.
            for slot in range(len(hits), neighbours):
                vertex_index[index, slot] = hits[0][1]
            distance[index] = hits[0][2]
            found[index] = True
        yield stop / count

    mapping = _Mapping("VERTEX")
    weight, within = _distance_weights(distance, max_distance, falloff)
    mapping.valid = found & within
    mapping.weight = np.where(mapping.valid, weight, 0.0)
    mapping.distance = distance
    mapping.missed = int(count - int(np.count_nonzero(mapping.valid)))
    mapping.vertex_index = vertex_index
    mapping.vertex_weight = np.where(mapping.valid[:, None], vertex_weight, 0.0)
    return mapping


def _sample_source(mapping: _Mapping, source_points: np.ndarray) -> np.ndarray:
    if mapping.kind == "VERTEX":
        return (source_points[mapping.vertex_index] * mapping.vertex_weight[:, :, None]).sum(axis=1)

    tris = mapping.triangles
    corner_a = source_points[tris[:, 0]]
    corner_b = source_points[tris[:, 1]]
    corner_c = source_points[tris[:, 2]]
    index = mapping.triangle_index
    surface = (
        corner_a[index] * mapping.barycentric[:, 0:1]
        + corner_b[index] * mapping.barycentric[:, 1:2]
        + corner_c[index] * mapping.barycentric[:, 2:3]
    )
    frames = _triangle_frames(corner_a, corner_b, corner_c)
    return surface + np.einsum("nij,nj->ni", frames[index], mapping.local_offset)


# ---------------------------------------------------------------------------
# Post processing
# ---------------------------------------------------------------------------


def _edge_arrays(mesh):
    count = len(mesh.edges)
    if count == 0:
        return None
    flat = np.empty(count * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", flat)
    pairs = flat.reshape(count, 2)
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def _smooth_field(field: np.ndarray, edges, vertex_count: int, factor: float, iterations: int) -> np.ndarray:
    if edges is None or iterations <= 0 or factor <= 0.0:
        return field
    first, second = edges
    counts = np.bincount(first, minlength=vertex_count) + np.bincount(second, minlength=vertex_count)
    movable = counts > 0
    if not movable.any():
        return field
    safe_counts = np.maximum(counts, 1).astype(np.float64)

    for _ in range(iterations):
        average = np.empty_like(field)
        for axis in range(3):
            accumulated = np.bincount(first, weights=field[second, axis], minlength=vertex_count)
            accumulated += np.bincount(second, weights=field[first, axis], minlength=vertex_count)
            average[:, axis] = accumulated
        average /= safe_counts[:, None]
        field[movable] += factor * (average[movable] - field[movable])
    return field


def _mirror_partners(points: np.ndarray, threshold: float):
    count = len(points)
    tree = KDTree(count)
    for index, position in enumerate(points):
        tree.insert((float(position[0]), float(position[1]), float(position[2])), index)
    tree.balance()

    partner = np.arange(count, dtype=np.int32)
    matched = np.zeros(count, dtype=bool)
    for index, position in enumerate(points):
        mirrored = (-float(position[0]), float(position[1]), float(position[2]))
        _position, found_index, distance = tree.find(mirrored)
        if found_index is not None and distance <= threshold:
            partner[index] = found_index
            matched[index] = True
    return partner, matched


def _symmetrize_field(field: np.ndarray, partner: np.ndarray, matched: np.ndarray) -> np.ndarray:
    mirrored = field[partner].copy()
    mirrored[:, 0] *= -1.0
    blended = (field + mirrored) * 0.5
    field[matched] = blended[matched]
    return field


# ---------------------------------------------------------------------------
# Shape key writing
# ---------------------------------------------------------------------------


def _remove_all_shape_keys(obj):
    while obj.data.shape_keys and obj.data.shape_keys.key_blocks:
        obj.shape_key_remove(obj.data.shape_keys.key_blocks[-1])


def _copy_key_metadata(target_object, source_record, target_key, copy_values: bool, use_relative: bool):
    new_min = source_record["slider_min"]
    new_max = source_record["slider_max"]
    # Blender clamps slider_min against the current slider_max, so widen the
    # upper bound first whenever the incoming range sits above the existing one.
    if new_min > target_key.slider_max:
        target_key.slider_max = new_max
    target_key.slider_min = new_min
    target_key.slider_max = new_max

    target_key.mute = source_record["mute"]
    target_key.interpolation = source_record["interpolation"]

    vertex_group = source_record["vertex_group"]
    target_key.vertex_group = vertex_group if vertex_group and target_object.vertex_groups.get(vertex_group) else ""

    if use_relative:
        if copy_values:
            target_key.value = source_record["value"]
    else:
        # ShapeKey.frame is derived from the key order and is read-only on some
        # Blender builds; ordering already reproduces it.
        try:
            target_key.frame = source_record["frame"]
        except AttributeError:
            pass


def _transfer_order(records, record_by_name, writable, basis_name, use_relative):
    """Iterative dependency order so relative chains are written base-first."""
    if not use_relative:
        return [record for record in records if record["name"] in writable]

    ordered = []
    state = {}

    def relative_of(record):
        name = record["relative_name"]
        if name == record["name"] or name not in record_by_name:
            return None
        return name

    for record in records:
        if record["name"] not in writable:
            continue
        stack = [record["name"]]
        while stack:
            name = stack[-1]
            status = state.get(name)
            if status == "done":
                stack.pop()
                continue
            if status == "visiting":
                state[name] = "done"
                stack.pop()
                if name in writable:
                    ordered.append(record_by_name[name])
                continue
            state[name] = "visiting"
            parent = relative_of(record_by_name[name])
            if parent is not None and parent != basis_name and parent in writable and state.get(parent) is None:
                stack.append(parent)
    return ordered


def _prune_empty_keys(target, candidates: set[str]) -> int:
    """Drop empty keys, recomputing protection after every removal."""
    removed = 0
    while candidates:
        shape_keys = target.data.shape_keys
        if shape_keys is None:
            break
        protected = {
            key_block.relative_key.name
            for key_block in shape_keys.key_blocks[1:]
            if key_block.relative_key is not None and key_block.relative_key != key_block
        }
        victim = None
        for name in list(candidates):
            key_block = shape_keys.key_blocks.get(name)
            if key_block is None:
                candidates.discard(name)
                continue
            if name not in protected:
                victim = key_block
                candidates.discard(name)
                break
        if victim is None:
            break
        target.shape_key_remove(victim)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Mapping cache
# ---------------------------------------------------------------------------


def _cache_key(settings, source, target, source_points, target_points):
    return (
        source.name,
        target.name,
        hash(np.ascontiguousarray(source_points).tobytes()),
        hash(np.ascontiguousarray(target_points).tobytes()),
        round(settings.normal_weight, 6),
        settings.use_max_distance,
        round(settings.max_distance, 9),
        round(settings.distance_falloff, 6),
    )


def _cache_store(key, mapping):
    if len(_MAPPING_CACHE) >= _MAPPING_CACHE_LIMIT:
        _MAPPING_CACHE.pop(next(iter(_MAPPING_CACHE)))
    _MAPPING_CACHE[key] = mapping


# ---------------------------------------------------------------------------
# Transfer core
# ---------------------------------------------------------------------------


def _build_mapping_for_target(settings, source, target, source_points, target_points, source_mesh):
    """Generator. Yields progress in [0, 1] and returns (mapping, method, notes)."""
    notes = []
    # None means unlimited. A zero limit is taken literally rather than being
    # folded into "unlimited", which is what a 0.0 sentinel used to do.
    max_distance = settings.max_distance if settings.use_max_distance else None
    if max_distance is not None and max_distance <= 0.0:
        notes.append("The distance limit is zero; only vertices already touching the source can be mapped.")

    key = None
    if settings.use_cache:
        key = _cache_key(settings, source, target, source_points, target_points)
        cached = _MAPPING_CACHE.get(key)
        if cached is not None:
            yield 1.0
            cached_mapping, cached_method, cached_notes = cached
            return cached_mapping, cached_method, notes + list(cached_notes)

    triangles = _source_triangles(source_mesh)
    if triangles is not None:
        target_normals = _vertex_normals(target.data, len(target_points))
        if target_normals is not None:
            target_normals = _transform_directions_normalized(target_normals, target.matrix_world)
        else:
            notes.append("Target vertex normals are unavailable; plain nearest-surface search was used.")
        mapping = yield from _build_surface_mapping(
            source_points,
            triangles,
            target_points,
            target_normals,
            max_distance,
            settings.distance_falloff,
            settings.normal_weight,
        )
        if mapping is not None:
            if key is not None:
                _cache_store(key, (mapping, "SURFACE", list(notes)))
            return mapping, "SURFACE", notes
        notes.append("The source has no usable faces; nearest-vertex mapping was used instead.")
    else:
        notes.append("The source has no faces; nearest-vertex mapping was used instead.")

    mapping = yield from _build_vertex_mapping(
        source_points, target_points, max_distance, settings.distance_falloff
    )
    if key is not None:
        _cache_store(key, (mapping, "VERTEX", list(notes)))
    return mapping, "VERTEX", notes


def _transform_directions_normalized(directions: np.ndarray, matrix) -> np.ndarray:
    # Normals transform by the inverse transpose of the 3x3 basis.
    basis = _matrix3(matrix)
    try:
        normal_matrix = np.linalg.inv(basis).T
    except np.linalg.LinAlgError:
        normal_matrix = basis
    transformed = directions @ normal_matrix.T
    lengths = np.maximum(np.linalg.norm(transformed, axis=1), 1.0e-24)
    return transformed / lengths[:, None]


def _gather_target_points(target, settings, depsgraph):
    """Basis coordinates the target will actually have once the transfer starts."""
    mesh = target.data
    # 2-1: when existing keys are cleared, the new Basis is built from
    # mesh.vertices, so the mapping must be built against those coordinates.
    points = _mesh_basis_coords(mesh, ignore_shape_keys=settings.clear_target)
    if settings.use_evaluated and depsgraph is not None:
        evaluated = _evaluated_coords(target, depsgraph, len(mesh.vertices), len(mesh.polygons))
        if evaluated is not None:
            return points, evaluated
    return points, points


def _transfer_to_target(settings, source, target, snapshot, depsgraph):
    """Generator. Yields progress in [0, 1] and returns a per-target summary."""
    notes = []
    records = snapshot["records"]
    basis_record = records[0]
    use_relative = snapshot["use_relative"]

    if len(source.data.vertices) != snapshot["vertex_count"]:
        raise RuntimeError("The source vertex count changed during the transfer.")

    source_mesh = source.data
    source_basis_local = basis_record["coords"]
    if settings.use_evaluated and depsgraph is not None:
        evaluated = _evaluated_coords(source, depsgraph, len(source_mesh.vertices), len(source_mesh.polygons))
        if evaluated is not None:
            source_basis_local = evaluated
        else:
            notes.append("Source modifiers change the vertex layout; the unevaluated mesh was used.")

    target_basis_local, target_match_local = _gather_target_points(target, settings, depsgraph)
    if settings.use_evaluated and depsgraph is not None and target_match_local is target_basis_local:
        notes.append("Target modifiers change the vertex layout; the unevaluated mesh was used.")

    source_world = _transform_points(source_basis_local, source.matrix_world)
    target_world = _transform_points(target_match_local, target.matrix_world)

    mapping, resolved_method, mapping_notes = yield from _build_mapping_for_target(
        settings, source, target, source_world, target_world, source_mesh
    )
    notes.extend(mapping_notes)

    # --- from here on the target data is modified; no yields until done ---

    target_count = len(target.data.vertices)
    weights = mapping.weight.copy()

    mask = _vertex_group_weights(target, settings.mask_vertex_group)
    if settings.mask_vertex_group and mask is None:
        notes.append("The mask vertex group was not found on the target.")
    elif mask is not None:
        if settings.invert_mask:
            mask = 1.0 - mask
        weights *= mask
    weights *= settings.strength

    world_to_target = _matrix3(target.matrix_world.inverted_safe())

    edges = _edge_arrays(target.data) if settings.smooth_iterations > 0 else None
    partner = matched = None
    if settings.use_symmetry:
        partner, matched = _mirror_partners(target_basis_local, settings.symmetry_threshold)

    # Shape keys live on Mesh data. Make the target single-user so transferring
    # never modifies unselected linked objects.
    if target.data.users > 1:
        target.data = target.data.copy()

    previous_active_name = target.active_shape_key.name if target.active_shape_key else ""
    if settings.clear_target:
        _remove_all_shape_keys(target)

    if target.data.shape_keys is None:
        target_basis_key = target.shape_key_add(name=basis_record["name"], from_mix=False)
    else:
        target_basis_key = target.data.shape_keys.key_blocks[0]

    target_shape_keys = target.data.shape_keys
    target_shape_keys.use_relative = use_relative
    if not use_relative:
        target_shape_keys.eval_time = snapshot["eval_time"]

    selected_records = [record for record in records[1:] if _key_selected(record["name"], settings.key_filter)]
    if not selected_records:
        notes.append("No source shape key matched the name filter.")

    target_by_source_name = {basis_record["name"]: target_basis_key}
    writable = set()
    created_count = 0
    updated_count = 0

    for record in selected_records:
        existing = target_shape_keys.key_blocks.get(record["name"])
        if existing is not None:
            target_by_source_name[record["name"]] = existing
            if settings.overwrite_existing or settings.clear_target:
                writable.add(record["name"])
                updated_count += 1
        else:
            target_by_source_name[record["name"]] = target.shape_key_add(name=record["name"], from_mix=False)
            writable.add(record["name"])
            created_count += 1

    record_by_name = {record["name"]: record for record in records}

    for record in selected_records:
        if record["name"] not in writable:
            continue
        target_key = target_by_source_name[record["name"]]
        if use_relative:
            target_key.relative_key = target_by_source_name.get(record["relative_name"], target_basis_key)
        _copy_key_metadata(target, record, target_key, settings.copy_values, use_relative)

    ordered = _transfer_order(selected_records, record_by_name, writable, basis_record["name"], use_relative)

    def resolve_relative(record) -> str:
        name = record["relative_name"] if use_relative else basis_record["name"]
        if name == record["name"] or name not in record_by_name:
            return basis_record["name"]
        return name

    # Only shapes used as a relative base are worth keeping resident; caching
    # every key would hold one full coordinate array per shape key.
    base_names = {resolve_relative(record) for record in ordered}
    sampled_cache: dict[str, np.ndarray] = {}

    def sampled(name: str) -> np.ndarray:
        cached = sampled_cache.get(name)
        if cached is not None:
            return cached
        world = _transform_points(record_by_name[name]["coords"], source.matrix_world)
        positions = _sample_source(mapping, world)
        if name in base_names:
            sampled_cache[name] = positions
        return positions

    maximum_delta: dict[str, float] = {}

    for record in ordered:
        name = record["name"]
        relative_name = resolve_relative(record)

        delta_world = sampled(name) - sampled(relative_name)
        delta = delta_world @ world_to_target.T
        delta *= weights[:, None]

        if partner is not None:
            delta = _symmetrize_field(delta, partner, matched)
        if edges is not None:
            delta = _smooth_field(delta, edges, target_count, settings.smooth_factor, settings.smooth_iterations)

        target_key = target_by_source_name[name]
        target_relative_key = target_key.relative_key if use_relative else target_basis_key
        if target_relative_key is None:
            target_relative_key = target_basis_key
        base = _vector_array(target_relative_key.data, target_count)

        output = base + delta
        target_key.data.foreach_set("co", output.astype(np.float32).ravel())
        maximum_delta[name] = float(np.max(np.einsum("ij,ij->i", delta, delta))) if target_count else 0.0

    removed_count = 0
    if settings.remove_empty:
        threshold_squared = settings.empty_threshold * settings.empty_threshold
        # Blender may have renamed a key on creation, so prune by the key's own
        # name rather than the source name.
        candidates = {
            target_by_source_name[name].name
            for name, value in maximum_delta.items()
            if value <= threshold_squared and name in target_by_source_name
        }
        removed_count = _prune_empty_keys(target, candidates)

    if not settings.clear_target and previous_active_name and target.data.shape_keys:
        previous_index = target.data.shape_keys.key_blocks.find(previous_active_name)
        target.active_shape_key_index = max(0, previous_index)
    else:
        target.active_shape_key_index = 0

    target.data.update()

    return {
        "target": target.name,
        "method": resolved_method,
        "created": created_count,
        "updated": updated_count,
        "removed": removed_count,
        "missed": mapping.missed,
        "notes": notes,
    }


def _run_transfer(context, settings, source, targets, summaries):
    """Generator. Yields progress in [0, 1] and fills `summaries` as it goes.

    Results are appended to the caller's list so a cancelled run can still
    report the targets that finished.
    """
    snapshot = _snapshot_source(source)

    total = len(targets)
    for position, target in enumerate(targets):
        base = position / total
        span = 1.0 / total
        # Re-evaluate per target: writing shape keys invalidates the previous
        # evaluated state.
        depsgraph = context.evaluated_depsgraph_get() if settings.use_evaluated else None
        generator = _transfer_to_target(settings, source, target, snapshot, depsgraph)
        try:
            while True:
                fraction = next(generator)
                yield base + fraction * span
        except StopIteration as stop:
            summaries.append(stop.value)
        yield base + span

    return summaries


def _format_summary(summaries) -> tuple[str, list[str]]:
    transferred = sum(item["created"] + item["updated"] for item in summaries)
    removed = sum(item["removed"] for item in summaries)
    missed = sum(item["missed"] for item in summaries)
    message = _tr("Transferred {count} shape keys to {targets} meshes ({removed} empty keys removed).").format(
        count=transferred,
        targets=len(summaries),
        removed=removed,
    )
    warnings = []
    if missed:
        warnings.append(_tr("{count} target vertices were outside the distance limit.").format(count=missed))
    seen = set()
    for item in summaries:
        for note in item["notes"]:
            if note not in seen:
                seen.add(note)
                warnings.append(_tr(note))
    return message, warnings


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _source_poll(_self, obj):
    return obj.type == "MESH"


class TOPU_FST_Settings(PropertyGroup):
    source_override: PointerProperty(
        name="Source",
        description="Explicit source mesh. Leave empty to use the two-object selection rule; set it to transfer to every other selected mesh at once",
        type=Object,
        poll=_source_poll,
    )
    normal_weight: FloatProperty(
        name="Normal Weight",
        description="How strongly normal disagreement penalises a candidate source triangle. Raise it when vertices snap across lips, eyelids or thin panels",
        default=0.5,
        min=0.0,
        max=4.0,
    )
    use_evaluated: BoolProperty(
        name="Use Modifier Result",
        description="Match against the modifier-evaluated shape. Ignored when a modifier changes the vertex or face count",
        default=False,
    )
    clear_target: BoolProperty(
        name="Remove Existing Target Shape Keys",
        description="Remove every shape key on the target before transfer",
        default=True,
    )
    overwrite_existing: BoolProperty(
        name="Overwrite Matching Shape Keys",
        description="Update target shape keys whose names match source shape keys",
        default=True,
    )
    key_filter: StringProperty(
        name="Name Filter",
        description="Comma separated wildcard patterns. Only matching source shape keys are transferred. Leave empty for all",
        default="",
    )
    strength: FloatProperty(
        name="Strength",
        description="Overall multiplier applied to every transferred deformation",
        default=1.0,
        min=-2.0,
        soft_min=0.0,
        soft_max=2.0,
    )
    mask_vertex_group: StringProperty(
        name="Mask Group",
        description="Target vertex group used to weight the transfer. Leave empty to transfer everywhere",
        default="",
    )
    invert_mask: BoolProperty(
        name="Invert Mask",
        description="Invert the mask vertex group weights",
        default=False,
    )
    remove_empty: BoolProperty(
        name="Remove Empty Shape Keys",
        description="Remove transferred keys whose deformation is below the empty threshold, except keys used as relative bases",
        default=True,
    )
    empty_threshold: FloatProperty(
        name="Empty Threshold",
        description="Maximum deformation length considered empty in target local space",
        default=0.000001,
        min=0.0,
        soft_max=0.01,
        precision=6,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    use_max_distance: BoolProperty(
        name="Limit Transfer Distance",
        description="Leave target vertices unchanged when no source surface or vertex is within the specified world-space distance",
        default=False,
    )
    max_distance: FloatProperty(
        name="Maximum Distance",
        description="Maximum world-space distance used for mapping",
        default=0.1,
        min=0.0,
        soft_max=10.0,
        precision=4,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    distance_falloff: FloatProperty(
        name="Distance Falloff",
        description="Fraction of the maximum distance over which the transfer fades out, preventing a hard seam at the limit",
        default=0.25,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    smooth_iterations: IntProperty(
        name="Smooth Iterations",
        description="Laplacian smoothing passes applied to the transferred displacement. Reduces faceting when the source is coarser than the target",
        default=0,
        min=0,
        max=50,
    )
    smooth_factor: FloatProperty(
        name="Smooth Factor",
        description="Strength of each smoothing pass",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    use_symmetry: BoolProperty(
        name="Symmetrize Result",
        description="Average each transferred displacement with its mirrored counterpart across the target local X axis",
        default=False,
    )
    symmetry_threshold: FloatProperty(
        name="Symmetry Threshold",
        description="Maximum distance between mirrored target vertices considered a symmetric pair",
        default=0.0001,
        min=0.0,
        soft_max=0.01,
        precision=5,
        subtype="DISTANCE",
        unit="LENGTH",
    )
    copy_values: BoolProperty(
        name="Copy Current Values",
        description="Copy each relative shape key's current value from the source",
        default=False,
    )
    use_modal: BoolProperty(
        name="Interactive Run",
        description="Run the mapping stage in the background so Esc can cancel it before anything is written",
        default=True,
    )
    use_cache: BoolProperty(
        name="Cache Mapping",
        description="Reuse the computed mapping while the meshes and mapping options are unchanged",
        default=True,
    )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TOPU_OT_forced_shape_key_transfer(Operator):
    bl_idname = "topu.forced_shape_key_transfer"
    bl_label = "Forced Shape Key Transfer"
    bl_description = "Transfer all source shape keys to the selected meshes even when topology differs"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        settings = _settings(context)
        if settings is None:
            return False
        source, targets, _message = _resolve_source_targets(context, settings)
        return source is not None and bool(targets)

    def _start(self, context):
        settings = _settings(context)
        if settings is None:
            self.report({"ERROR"}, _tr("Re-enable the add-on to finish loading it."))
            return False
        source, targets, message = _resolve_source_targets(context, settings)
        if source is None or not targets:
            self.report({"ERROR"}, _tr(message))
            return False
        self._summaries = []
        self._generator = _run_transfer(context, settings, source, targets, self._summaries)
        self._window_manager = context.window_manager
        self._timer = None
        self._window_manager.progress_begin(0.0, 1.0)
        return True

    def _pump(self, deadline: float | None) -> bool:
        """Advance the generator. Returns False when the time slice expired."""
        while True:
            fraction = next(self._generator)
            self._window_manager.progress_update(fraction)
            if deadline is not None and time.perf_counter() >= deadline:
                return False

    def _report_result(self):
        message, warnings = _format_summary(self._summaries)
        for warning in warnings:
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, message)

    def _cleanup(self):
        window_manager = getattr(self, "_window_manager", None)
        if window_manager is None:
            return
        if getattr(self, "_timer", None) is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        window_manager.progress_end()
        self._window_manager = None

    def execute(self, context):
        if not self._start(context):
            return {"CANCELLED"}
        try:
            try:
                self._pump(None)
            except StopIteration:
                pass
            self._report_result()
            return {"FINISHED"}
        except Exception as error:  # Blender needs the full traceback in its console.
            traceback.print_exc()
            self.report({"ERROR"}, _tr("Shape key transfer failed: {error}").format(error=error))
            return {"CANCELLED"}
        finally:
            self._cleanup()

    def invoke(self, context, event):
        settings = _settings(context)
        if settings is None or not settings.use_modal:
            return self.execute(context)
        if not self._start(context):
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(0.02, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            # A yield only ever happens while the mapping is being built or
            # between targets, so nothing is half-written at this point.
            self._generator.close()
            self._cleanup()
            if self._summaries:
                self._report_result()
                self.report({"WARNING"}, _tr("Cancelled; remaining targets were left untouched."))
                return {"FINISHED"}
            self.report({"WARNING"}, _tr("Shape key transfer cancelled."))
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        try:
            if not self._pump(time.perf_counter() + 0.05):
                return {"RUNNING_MODAL"}
        except StopIteration:
            self._report_result()
            self._cleanup()
            return {"FINISHED"}
        except Exception as error:
            traceback.print_exc()
            self._cleanup()
            self.report({"ERROR"}, _tr("Shape key transfer failed: {error}").format(error=error))
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}


class TOPU_OT_analyze_shape_key_transfer(Operator):
    bl_idname = "topu.analyze_shape_key_transfer"
    bl_label = "Analyze Mapping"
    bl_description = "Build the mapping without writing anything and report match distances, to help pick a distance limit"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return TOPU_OT_forced_shape_key_transfer.poll(context)

    def execute(self, context):
        settings = _settings(context)
        if settings is None:
            self.report({"ERROR"}, _tr("Re-enable the add-on to finish loading it."))
            return {"CANCELLED"}
        source, targets, message = _resolve_source_targets(context, settings)
        if source is None or not targets:
            self.report({"ERROR"}, _tr(message))
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get() if settings.use_evaluated else None
        source_mesh = source.data
        source_basis_local = _mesh_basis_coords(source_mesh)
        if settings.use_evaluated and depsgraph is not None:
            evaluated = _evaluated_coords(source, depsgraph, len(source_mesh.vertices), len(source_mesh.polygons))
            if evaluated is not None:
                source_basis_local = evaluated
        source_world = _transform_points(source_basis_local, source.matrix_world)

        lines = []
        try:
            for target in targets:
                _basis, match_local = _gather_target_points(target, settings, depsgraph)
                target_world = _transform_points(match_local, target.matrix_world)
                generator = _build_mapping_for_target(
                    settings, source, target, source_world, target_world, source_mesh
                )
                mapping = method = None
                try:
                    while True:
                        next(generator)
                except StopIteration as stop:
                    mapping, method, _notes = stop.value

                finite = mapping.distance[np.isfinite(mapping.distance)]
                if len(finite):
                    lines.append(
                        _tr("{target}: {method}, mean {mean:.4f} / max {maximum:.4f}, {missed} unmapped").format(
                            target=target.name,
                            method=method,
                            mean=float(np.mean(finite)),
                            maximum=float(np.max(finite)),
                            missed=mapping.missed,
                        )
                    )
                else:
                    lines.append(
                        _tr("{target}: {method}, no vertex could be mapped").format(target=target.name, method=method)
                    )
        except Exception as error:
            traceback.print_exc()
            self.report({"ERROR"}, _tr("Shape key transfer failed: {error}").format(error=error))
            return {"CANCELLED"}

        _LAST_ANALYSIS.clear()
        _LAST_ANALYSIS.extend(lines)
        self.report({"INFO"}, lines[0] if lines else _tr("No mapping was produced."))
        return {"FINISHED"}


class TOPU_OT_clear_mapping_cache(Operator):
    bl_idname = "topu.clear_shape_key_mapping_cache"
    bl_label = "Clear Mapping Cache"
    bl_description = "Discard cached mapping results"
    bl_options = {"REGISTER"}

    def execute(self, _context):
        _MAPPING_CACHE.clear()
        _LAST_ANALYSIS.clear()
        self.report({"INFO"}, _tr("Mapping cache cleared."))
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class DATA_PT_topu_forced_shape_key_transfer(Panel):
    bl_label = "Forced Shape Key Transfer"
    bl_idname = "DATA_PT_topu_forced_shape_key_transfer"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "data"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        settings = _settings(context)
        if settings is None:
            layout.label(text=_tr("Re-enable the add-on to finish loading it."), icon="ERROR")
            return

        layout.use_property_split = True
        layout.use_property_decorate = False
        source, targets, message = _resolve_source_targets(context, settings)

        info_box = layout.box()
        info_column = info_box.column(align=True)
        if source is not None and targets:
            info_column.label(text=_tr("Source: {name}").format(name=source.name), icon="SHAPEKEY_DATA")
            if len(targets) == 1:
                info_column.label(text=_tr("Target (Active): {name}").format(name=targets[0].name), icon="OBJECT_DATA")
            else:
                info_column.label(text=_tr("Targets: {count}").format(count=len(targets)), icon="OBJECT_DATA")
        else:
            info_column.label(text=_tr(message), icon="INFO")
            info_column.label(text=_tr("Select the source first, then Shift-select the target."))

        layout.prop(settings, "source_override")

        header, body = layout.panel("topu_fst_accuracy", default_closed=False)
        header.label(text=_tr("Accuracy"))
        if body is not None:
            body.prop(settings, "normal_weight")
            body.prop(settings, "use_evaluated")

        header, body = layout.panel("topu_fst_distance", default_closed=True)
        header.use_property_split = False
        header.prop(settings, "use_max_distance", text="")
        header.label(text=_tr("Limit Transfer Distance"))
        if body is not None:
            body.enabled = settings.use_max_distance
            body.prop(settings, "max_distance")
            body.prop(settings, "distance_falloff")

        header, body = layout.panel("topu_fst_filter", default_closed=True)
        header.label(text=_tr("Selection And Weighting"))
        if body is not None:
            body.prop(settings, "key_filter")
            body.prop(settings, "strength")
            body.prop_search(settings, "mask_vertex_group", context.object, "vertex_groups")
            mask_row = body.row()
            mask_row.enabled = bool(settings.mask_vertex_group)
            mask_row.prop(settings, "invert_mask")

        header, body = layout.panel("topu_fst_postprocess", default_closed=True)
        header.label(text=_tr("Post Processing"))
        if body is not None:
            body.prop(settings, "smooth_iterations")
            smooth_row = body.row()
            smooth_row.enabled = settings.smooth_iterations > 0
            smooth_row.prop(settings, "smooth_factor")
            body.prop(settings, "use_symmetry")
            symmetry_row = body.row()
            symmetry_row.enabled = settings.use_symmetry
            symmetry_row.prop(settings, "symmetry_threshold")

        header, body = layout.panel("topu_fst_output", default_closed=False)
        header.label(text=_tr("Output"))
        if body is not None:
            clear_row = body.row()
            clear_row.alert = settings.clear_target
            clear_row.prop(settings, "clear_target")
            overwrite_row = body.row()
            overwrite_row.enabled = not settings.clear_target
            overwrite_row.prop(settings, "overwrite_existing")
            body.prop(settings, "copy_values")
            body.prop(settings, "remove_empty")
            threshold_row = body.row()
            threshold_row.enabled = settings.remove_empty
            threshold_row.prop(settings, "empty_threshold")

        header, body = layout.panel("topu_fst_performance", default_closed=True)
        header.label(text=_tr("Performance"))
        if body is not None:
            body.prop(settings, "use_modal")
            body.prop(settings, "use_cache")
            body.operator(TOPU_OT_clear_mapping_cache.bl_idname, icon="TRASH")

        layout.separator()
        analyze_row = layout.row()
        analyze_row.enabled = source is not None and bool(targets)
        analyze_row.operator(TOPU_OT_analyze_shape_key_transfer.bl_idname, icon="VIEWZOOM")
        if _LAST_ANALYSIS:
            analysis_box = layout.box().column(align=True)
            for line in _LAST_ANALYSIS[:8]:
                analysis_box.label(text=line)

        execute_row = layout.row()
        execute_row.scale_y = 1.4
        execute_row.enabled = source is not None and bool(targets)
        execute_row.operator(TOPU_OT_forced_shape_key_transfer.bl_idname, icon="MOD_DATA_TRANSFER")


def _draw_shape_key_menu(self, _context):
    self.layout.separator()
    self.layout.operator(TOPU_OT_forced_shape_key_transfer.bl_idname, icon="MOD_DATA_TRANSFER")


TRANSLATIONS = {
    "ja_JP": {
        ("*", "Forced Shape Key Transfer"): "シェイプキー強制転送",
        ("*", "Transfer all source shape keys to the selected meshes even when topology differs"): "トポロジーが異なるメッシュ間でも、参照元の全シェイプキーを選択中のメッシュへ転送します",
        ("*", "Source"): "参照元",
        ("*", "Explicit source mesh. Leave empty to use the two-object selection rule; set it to transfer to every other selected mesh at once"): "参照元メッシュを明示指定します。空の場合は2オブジェクト選択の規則を使い、指定した場合は他の選択メッシュすべてへ一括転送します",
        ("*", "Accuracy"): "精度",
        ("*", "Normal Weight"): "法線の重み",
        ("*", "How strongly normal disagreement penalises a candidate source triangle. Raise it when vertices snap across lips, eyelids or thin panels"): "法線の不一致を候補面の評価にどれだけ反映するかです。唇・まぶた・薄板の裏側へ吸着する場合は上げてください",
        ("*", "Use Modifier Result"): "モディファイア適用後で対応付け",
        ("*", "Match against the modifier-evaluated shape. Ignored when a modifier changes the vertex or face count"): "モディファイア評価後の形状で対応付けます。頂点数や面数が変わる場合は無視されます",
        ("*", "Remove Existing Target Shape Keys"): "転送先の既存シェイプキーを削除",
        ("*", "Remove every shape key on the target before transfer"): "転送前に転送先の全シェイプキーを削除します",
        ("*", "Overwrite Matching Shape Keys"): "同名シェイプキーを上書き",
        ("*", "Update target shape keys whose names match source shape keys"): "参照元と同名の転送先シェイプキーを更新します",
        ("*", "Selection And Weighting"): "対象と重み付け",
        ("*", "Name Filter"): "名前フィルタ",
        ("*", "Comma separated wildcard patterns. Only matching source shape keys are transferred. Leave empty for all"): "カンマ区切りのワイルドカードです。一致した参照元シェイプキーのみ転送します。空の場合は全て転送します",
        ("*", "Strength"): "強度",
        ("*", "Overall multiplier applied to every transferred deformation"): "転送する全変形に掛かる倍率です",
        ("*", "Mask Group"): "マスク頂点グループ",
        ("*", "Target vertex group used to weight the transfer. Leave empty to transfer everywhere"): "転送の重みに使う転送先の頂点グループです。空の場合は全体に転送します",
        ("*", "Invert Mask"): "マスクを反転",
        ("*", "Invert the mask vertex group weights"): "マスク頂点グループのウェイトを反転します",
        ("*", "Remove Empty Shape Keys"): "変形のないシェイプキーを削除",
        ("*", "Remove transferred keys whose deformation is below the empty threshold, except keys used as relative bases"): "相対基準として使用されているキーを除き、変形量がしきい値以下の転送済みキーを削除します",
        ("*", "Empty Threshold"): "空判定しきい値",
        ("*", "Maximum deformation length considered empty in target local space"): "転送先ローカル空間で変形なしと判定する最大変形量です",
        ("*", "Limit Transfer Distance"): "転送距離を制限",
        ("*", "Leave target vertices unchanged when no source surface or vertex is within the specified world-space distance"): "指定したワールド空間距離内に参照元がない転送先頂点は変形させません",
        ("*", "Maximum Distance"): "最大距離",
        ("*", "Maximum world-space distance used for mapping"): "転送の対応付けに使用するワールド空間上の最大距離です",
        ("*", "Distance Falloff"): "距離の減衰",
        ("*", "Fraction of the maximum distance over which the transfer fades out, preventing a hard seam at the limit"): "最大距離のうち転送量を滑らかに減衰させる割合です。制限境界での段差を防ぎます",
        ("*", "Post Processing"): "後処理",
        ("*", "Smooth Iterations"): "スムージング回数",
        ("*", "Laplacian smoothing passes applied to the transferred displacement. Reduces faceting when the source is coarser than the target"): "転送後の変位に掛けるラプラシアン平滑化の回数です。参照元が粗い場合のカクつきを軽減します",
        ("*", "Smooth Factor"): "スムージング強度",
        ("*", "Strength of each smoothing pass"): "1回あたりの平滑化の強さです",
        ("*", "Symmetrize Result"): "結果を左右対称化",
        ("*", "Average each transferred displacement with its mirrored counterpart across the target local X axis"): "転送先ローカルX軸で対になる頂点の変位と平均化します",
        ("*", "Symmetry Threshold"): "対称判定しきい値",
        ("*", "Maximum distance between mirrored target vertices considered a symmetric pair"): "左右対称な頂点ペアとみなす最大距離です",
        ("*", "Copy Current Values"): "現在値もコピー",
        ("*", "Copy each relative shape key's current value from the source"): "相対シェイプキーの現在値も参照元からコピーします",
        ("*", "Output"): "出力",
        ("*", "Performance"): "パフォーマンス",
        ("*", "Interactive Run"): "対話実行",
        ("*", "Run the mapping stage in the background so Esc can cancel it before anything is written"): "対応付けを段階実行し、書き込み前ならEscで中断できるようにします",
        ("*", "Cache Mapping"): "対応付けをキャッシュ",
        ("*", "Reuse the computed mapping while the meshes and mapping options are unchanged"): "メッシュと対応付け設定が変わらない間、計算済みの対応付けを再利用します",
        ("*", "Clear Mapping Cache"): "対応付けキャッシュを破棄",
        ("*", "Discard cached mapping results"): "キャッシュ済みの対応付け結果を破棄します",
        ("*", "Mapping cache cleared."): "対応付けキャッシュを破棄しました。",
        ("*", "Analyze Mapping"): "対応付けを解析",
        ("*", "Build the mapping without writing anything and report match distances, to help pick a distance limit"): "書き込みを行わずに対応付けだけを構築し、距離制限の目安として一致距離を表示します",
        ("*", "Source: {name}"): "参照元: {name}",
        ("*", "Target (Active): {name}"): "転送先（アクティブ）: {name}",
        ("*", "Targets: {count}"): "転送先: {count}個",
        ("*", "Select the source first, then Shift-select the target."): "参照元を選び、続けてShiftで転送先をアクティブ選択してください。",
        ("*", "The active object must be a mesh."): "アクティブオブジェクトをメッシュにしてください。",
        ("*", "Switch to Object Mode before transferring."): "オブジェクトモードに切り替えてから実行してください。",
        ("*", "Select exactly two mesh objects, or pick an explicit source."): "メッシュオブジェクトを2つだけ選択するか、参照元を明示指定してください。",
        ("*", "Select at least one target mesh besides the source."): "参照元以外に転送先メッシュを1つ以上選択してください。",
        ("*", "The source and target must both be mesh objects."): "参照元と転送先は両方ともメッシュである必要があります。",
        ("*", "The source has no transferable shape keys."): "参照元に転送可能なシェイプキーがありません。",
        ("*", "The source and target meshes must contain vertices."): "参照元と転送先のメッシュには頂点が必要です。",
        ("*", "The source has no faces; nearest-vertex mapping was used instead."): "参照元に面がないため、最近傍頂点方式へ切り替えました。",
        ("*", "The source has no usable faces; nearest-vertex mapping was used instead."): "参照元に使用可能な面がないため、最近傍頂点方式へ切り替えました。",
        ("*", "Target vertex normals are unavailable; plain nearest-surface search was used."): "転送先の頂点法線を取得できないため、単純な最近傍探索を使用しました。",
        ("*", "Source modifiers change the vertex layout; the unevaluated mesh was used."): "参照元のモディファイアが頂点構成を変えるため、未適用のメッシュを使用しました。",
        ("*", "Target modifiers change the vertex layout; the unevaluated mesh was used."): "転送先のモディファイアが頂点構成を変えるため、未適用のメッシュを使用しました。",
        ("*", "The mask vertex group was not found on the target."): "マスク用の頂点グループが転送先に見つかりませんでした。",
        ("*", "No source shape key matched the name filter."): "名前フィルタに一致する参照元シェイプキーがありません。",
        ("*", "The distance limit is zero; only vertices already touching the source can be mapped."): "距離制限が0のため、参照元に接している頂点しか対応付けできません。",
        ("*", "Transferred {count} shape keys to {targets} meshes ({removed} empty keys removed)."): "{count}個のシェイプキーを{targets}個のメッシュへ転送しました（空のキーを{removed}個削除）。",
        ("*", "{count} target vertices were outside the distance limit."): "転送先の{count}頂点が距離制限外でした。",
        ("*", "{target}: {method}, mean {mean:.4f} / max {maximum:.4f}, {missed} unmapped"): "{target}: {method}、平均{mean:.4f} / 最大{maximum:.4f}、未対応{missed}頂点",
        ("*", "{target}: {method}, no vertex could be mapped"): "{target}: {method}、対応付けできた頂点がありません",
        ("*", "No mapping was produced."): "対応付けを作成できませんでした。",
        ("*", "Cancelled; remaining targets were left untouched."): "中断しました。未処理の転送先は変更していません。",
        ("*", "Shape key transfer cancelled."): "シェイプキー転送を中断しました。",
        ("*", "Re-enable the add-on to finish loading it."): "アドオンを無効化してから再度有効化してください。",
        ("*", "Shape key transfer failed: {error}"): "シェイプキー転送に失敗しました: {error}",
        ("*", "The source vertex count changed during the transfer."): "転送中に参照元の頂点数が変化しました。",
    }
}


def _mirror_operator_labels(translations):
    """Give every Operator bl_label a second entry under the operator context.

    Blender resolves an Operator's label through its own translation context
    ("Operator"), not the default one, so a menu or button label registered only
    under "*" stays in English while the identical Panel label is translated.
    """
    operator_context = bpy.app.translations.contexts.operator_default
    labels = [cls.bl_label for cls in CLASSES if issubclass(cls, Operator)]
    for entries in translations.values():
        for label in labels:
            translated = entries.get(("*", label))
            if translated is not None:
                entries.setdefault((operator_context, label), translated)
    return translations


CLASSES = (
    TOPU_FST_Settings,
    TOPU_OT_forced_shape_key_transfer,
    TOPU_OT_analyze_shape_key_transfer,
    TOPU_OT_clear_mapping_cache,
    DATA_PT_topu_forced_shape_key_transfer,
)

_registered_menus = []


def _unregister_by_name(cls):
    """Drop whatever class is registered under this class's name.

    Reloading after the files changed on disk leaves Blender holding the
    previous module's classes. Those are different objects from the ones in
    CLASSES, so unregister_class() on ours raises and aborts the loop, stranding
    a live panel that then draws against settings we already removed.
    """
    existing = getattr(bpy.types, cls.__name__, None)
    for candidate in (existing, cls):
        if candidate is None:
            continue
        try:
            bpy.utils.unregister_class(candidate)
            return
        except (RuntimeError, ValueError):
            continue


def register():
    try:
        for cls in CLASSES:
            _unregister_by_name(cls)
            bpy.utils.register_class(cls)
        bpy.types.Scene.topu_fst_settings = PointerProperty(type=TOPU_FST_Settings)

        try:
            bpy.app.translations.unregister(ADDON_ID)
        except Exception:
            pass
        bpy.app.translations.register(ADDON_ID, _mirror_operator_labels(TRANSLATIONS))

        for menu_name in ("MESH_MT_shape_key_context_menu", "MESH_MT_shape_key_specials"):
            menu_type = getattr(bpy.types, menu_name, None)
            if menu_type is not None:
                menu_type.append(_draw_shape_key_menu)
                _registered_menus.append(menu_type)
    except Exception:
        # A half-registered add-on raises on every redraw. Undo everything and
        # let the failure surface instead.
        unregister()
        raise


def unregister():
    _MAPPING_CACHE.clear()
    _LAST_ANALYSIS.clear()

    for menu_type in reversed(_registered_menus):
        try:
            menu_type.remove(_draw_shape_key_menu)
        except Exception:
            pass
    _registered_menus.clear()

    try:
        bpy.app.translations.unregister(ADDON_ID)
    except Exception:
        pass

    # Classes go first: a panel that is still registered can be drawn, and it
    # needs the scene property to exist.
    for cls in reversed(CLASSES):
        _unregister_by_name(cls)

    if hasattr(bpy.types.Scene, "topu_fst_settings"):
        del bpy.types.Scene.topu_fst_settings


if __name__ == "__main__":
    register()
