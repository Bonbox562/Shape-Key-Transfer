# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 ToPu
#
# The transfer workflow was independently reimplemented after studying
# Blender-CM3D2-Converter, licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.

"""Transfer relative or absolute shape keys between different mesh topologies."""

from __future__ import annotations

from array import array
import traceback

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


bl_info = {
    "name": "ToPu Forced Shape Key Transfer",
    "author": "ToPu",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Object Data Properties > Forced Shape Key Transfer",
    "description": "Transfer shape keys between meshes with different topology",
    "category": "Object",
}


ADDON_ID = "topu_forced_shape_key_transfer"


def _tr(text: str) -> str:
    return bpy.app.translations.pgettext_iface(text)


def _resolve_source_target(context):
    target = context.active_object
    selected = list(context.selected_objects)

    if target is None or target.type != "MESH":
        return None, None, "The active object must be a mesh."
    if target.mode != "OBJECT":
        return None, None, "Switch to Object Mode before transferring."
    if len(selected) != 2:
        return None, None, "Select exactly two mesh objects."

    source_candidates = [obj for obj in selected if obj != target]
    if len(source_candidates) != 1 or source_candidates[0].type != "MESH":
        return None, None, "The source and target must both be mesh objects."

    source = source_candidates[0]
    source_keys = source.data.shape_keys
    if source_keys is None or len(source_keys.key_blocks) < 2:
        return None, None, "The source has no transferable shape keys."
    if len(source.data.vertices) == 0 or len(target.data.vertices) == 0:
        return None, None, "The source and target meshes must contain vertices."

    return source, target, ""


def _collection_coords(points, count: int) -> array:
    coords = array("f", [0.0]) * (count * 3)
    points.foreach_get("co", coords)
    return coords


def _key_coords(key_block, count: int) -> array:
    return _collection_coords(key_block.data, count)


def _mesh_basis_coords(mesh) -> array:
    count = len(mesh.vertices)
    if mesh.shape_keys and mesh.shape_keys.key_blocks:
        return _key_coords(mesh.shape_keys.key_blocks[0], count)
    return _collection_coords(mesh.vertices, count)


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
                "coords": _key_coords(key_block, vertex_count),
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


def _world_positions(coords: array, matrix_world) -> list[Vector]:
    return [
        matrix_world @ Vector((coords[index], coords[index + 1], coords[index + 2]))
        for index in range(0, len(coords), 3)
    ]


def _barycentric_weights(point: Vector, a: Vector, b: Vector, c: Vector):
    edge_0 = b - a
    edge_1 = c - a
    point_vector = point - a
    dot_00 = edge_0.dot(edge_0)
    dot_01 = edge_0.dot(edge_1)
    dot_11 = edge_1.dot(edge_1)
    dot_20 = point_vector.dot(edge_0)
    dot_21 = point_vector.dot(edge_1)
    denominator = dot_00 * dot_11 - dot_01 * dot_01

    if abs(denominator) <= 1.0e-20:
        distances = ((point - a).length_squared, (point - b).length_squared, (point - c).length_squared)
        nearest = min(range(3), key=distances.__getitem__)
        return (1.0, 0.0, 0.0) if nearest == 0 else ((0.0, 1.0, 0.0) if nearest == 1 else (0.0, 0.0, 1.0))

    weight_b = (dot_11 * dot_20 - dot_01 * dot_21) / denominator
    weight_c = (dot_00 * dot_21 - dot_01 * dot_20) / denominator
    weight_a = 1.0 - weight_b - weight_c

    # find_nearest returns a point on the triangle. Clamp tiny floating-point
    # excursions so interpolated deltas remain stable at edges and corners.
    weights = [max(0.0, min(1.0, value)) for value in (weight_a, weight_b, weight_c)]
    total = sum(weights)
    if total <= 1.0e-20:
        return 1.0, 0.0, 0.0
    return weights[0] / total, weights[1] / total, weights[2] / total


def _build_surface_mapping(source, source_world, target_world, max_distance: float):
    source.data.calc_loop_triangles()
    triangles = [tuple(loop_triangle.vertices) for loop_triangle in source.data.loop_triangles]
    if not triangles:
        return None

    tree = BVHTree.FromPolygons(source_world, triangles, all_triangles=True)
    mapping = []
    missed = 0

    for target_position in target_world:
        result = tree.find_nearest(target_position, max_distance) if max_distance > 0.0 else tree.find_nearest(target_position)
        if result is None or result[0] is None or result[2] is None:
            mapping.append(None)
            missed += 1
            continue

        nearest_position, _normal, triangle_index, _distance = result
        vertex_a, vertex_b, vertex_c = triangles[triangle_index]
        weight_a, weight_b, weight_c = _barycentric_weights(
            nearest_position,
            source_world[vertex_a],
            source_world[vertex_b],
            source_world[vertex_c],
        )
        mapping.append((vertex_a, vertex_b, vertex_c, weight_a, weight_b, weight_c))

    return mapping, missed


def _build_vertex_mapping(source_world, target_world, max_distance: float):
    tree = KDTree(len(source_world))
    for index, position in enumerate(source_world):
        tree.insert(position, index)
    tree.balance()

    mapping = []
    missed = 0
    for target_position in target_world:
        _position, source_index, distance = tree.find(target_position)
        if max_distance > 0.0 and distance > max_distance:
            mapping.append(None)
            missed += 1
        else:
            mapping.append((source_index,))
    return mapping, missed


def _remove_all_shape_keys(obj):
    while obj.data.shape_keys and obj.data.shape_keys.key_blocks:
        obj.shape_key_remove(obj.data.shape_keys.key_blocks[-1])


def _copy_key_metadata(target_object, source_record, target_key, copy_values: bool, use_relative: bool):
    target_key.slider_min = source_record["slider_min"]
    target_key.slider_max = source_record["slider_max"]
    target_key.mute = source_record["mute"]
    target_key.interpolation = source_record["interpolation"]

    vertex_group = source_record["vertex_group"]
    target_key.vertex_group = vertex_group if vertex_group and target_object.vertex_groups.get(vertex_group) else ""

    if use_relative:
        if copy_values:
            target_key.value = source_record["value"]
    else:
        target_key.frame = source_record["frame"]


def _write_transferred_key(
    source_record,
    source_relative_record,
    target_key,
    target_relative_key,
    mapping,
    mapping_method: str,
    source_to_target,
    target_count: int,
):
    source_coords = source_record["coords"]
    relative_coords = source_relative_record["coords"]
    target_relative_coords = _key_coords(target_relative_key, target_count)
    output_coords = array("f", target_relative_coords)
    max_delta_squared = 0.0

    row_0 = source_to_target[0]
    row_1 = source_to_target[1]
    row_2 = source_to_target[2]

    if mapping_method == "SURFACE":
        for target_index, item in enumerate(mapping):
            if item is None:
                continue
            index_a, index_b, index_c, weight_a, weight_b, weight_c = item
            offset_a = index_a * 3
            offset_b = index_b * 3
            offset_c = index_c * 3

            delta_x = (
                (source_coords[offset_a] - relative_coords[offset_a]) * weight_a
                + (source_coords[offset_b] - relative_coords[offset_b]) * weight_b
                + (source_coords[offset_c] - relative_coords[offset_c]) * weight_c
            )
            delta_y = (
                (source_coords[offset_a + 1] - relative_coords[offset_a + 1]) * weight_a
                + (source_coords[offset_b + 1] - relative_coords[offset_b + 1]) * weight_b
                + (source_coords[offset_c + 1] - relative_coords[offset_c + 1]) * weight_c
            )
            delta_z = (
                (source_coords[offset_a + 2] - relative_coords[offset_a + 2]) * weight_a
                + (source_coords[offset_b + 2] - relative_coords[offset_b + 2]) * weight_b
                + (source_coords[offset_c + 2] - relative_coords[offset_c + 2]) * weight_c
            )

            target_delta_x = row_0[0] * delta_x + row_0[1] * delta_y + row_0[2] * delta_z
            target_delta_y = row_1[0] * delta_x + row_1[1] * delta_y + row_1[2] * delta_z
            target_delta_z = row_2[0] * delta_x + row_2[1] * delta_y + row_2[2] * delta_z
            target_offset = target_index * 3
            output_coords[target_offset] += target_delta_x
            output_coords[target_offset + 1] += target_delta_y
            output_coords[target_offset + 2] += target_delta_z
            max_delta_squared = max(
                max_delta_squared,
                target_delta_x * target_delta_x + target_delta_y * target_delta_y + target_delta_z * target_delta_z,
            )
    else:
        for target_index, item in enumerate(mapping):
            if item is None:
                continue
            source_offset = item[0] * 3
            delta_x = source_coords[source_offset] - relative_coords[source_offset]
            delta_y = source_coords[source_offset + 1] - relative_coords[source_offset + 1]
            delta_z = source_coords[source_offset + 2] - relative_coords[source_offset + 2]
            target_delta_x = row_0[0] * delta_x + row_0[1] * delta_y + row_0[2] * delta_z
            target_delta_y = row_1[0] * delta_x + row_1[1] * delta_y + row_1[2] * delta_z
            target_delta_z = row_2[0] * delta_x + row_2[1] * delta_y + row_2[2] * delta_z
            target_offset = target_index * 3
            output_coords[target_offset] += target_delta_x
            output_coords[target_offset + 1] += target_delta_y
            output_coords[target_offset + 2] += target_delta_z
            max_delta_squared = max(
                max_delta_squared,
                target_delta_x * target_delta_x + target_delta_y * target_delta_y + target_delta_z * target_delta_z,
            )

    target_key.data.foreach_set("co", output_coords)
    return max_delta_squared


class TOPU_FST_Settings(PropertyGroup):
    mapping_method: EnumProperty(
        name="Mapping",
        description="How source deformation is sampled for each target vertex",
        items=(
            (
                "SURFACE",
                "Nearest Surface",
                "Use the nearest source triangle and barycentric interpolation for smooth, accurate transfer",
                "SNAP_FACE",
                0,
            ),
            (
                "VERTEX",
                "Nearest Vertex (Legacy)",
                "Copy deformation from the single nearest source vertex, matching the original workflow",
                "VERTEXSEL",
                1,
            ),
        ),
        default="SURFACE",
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
    copy_values: BoolProperty(
        name="Copy Current Values",
        description="Copy each relative shape key's current value from the source",
        default=False,
    )


class TOPU_OT_forced_shape_key_transfer(Operator):
    bl_idname = "topu.forced_shape_key_transfer"
    bl_label = "Forced Shape Key Transfer"
    bl_description = "Transfer all source shape keys to the active mesh even when topology differs"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        source, target, _message = _resolve_source_target(context)
        return source is not None and target is not None

    def execute(self, context):
        source, target, message = _resolve_source_target(context)
        if source is None or target is None:
            self.report({"ERROR"}, _tr(message))
            return {"CANCELLED"}

        settings = context.scene.topu_fst_settings
        window_manager = context.window_manager
        progress_started = False

        try:
            source_snapshot = _snapshot_source(source)
            source_records = source_snapshot["records"]
            source_basis_record = source_records[0]
            target_count = len(target.data.vertices)
            target_basis_before = _mesh_basis_coords(target.data)

            source_world = _world_positions(source_basis_record["coords"], source.matrix_world)
            target_world = _world_positions(target_basis_before, target.matrix_world)
            max_distance = settings.max_distance if settings.use_max_distance else 0.0

            mapping_method = settings.mapping_method
            if mapping_method == "SURFACE":
                mapping_result = _build_surface_mapping(source, source_world, target_world, max_distance)
                if mapping_result is None:
                    mapping_method = "VERTEX"
                    self.report({"WARNING"}, _tr("The source has no faces; nearest-vertex mapping was used instead."))
                    mapping, missed_count = _build_vertex_mapping(source_world, target_world, max_distance)
                else:
                    mapping, missed_count = mapping_result
            else:
                mapping, missed_count = _build_vertex_mapping(source_world, target_world, max_distance)

            # Shape keys live on Mesh data. Make the active target single-user
            # so transferring never modifies unselected linked objects.
            if target.data.users > 1:
                target.data = target.data.copy()

            previous_active_name = target.active_shape_key.name if target.active_shape_key else ""
            if settings.clear_target:
                _remove_all_shape_keys(target)

            if target.data.shape_keys is None:
                target_basis_key = target.shape_key_add(name=source_basis_record["name"], from_mix=False)
            else:
                target_basis_key = target.data.shape_keys.key_blocks[0]

            target_shape_keys = target.data.shape_keys
            target_shape_keys.use_relative = source_snapshot["use_relative"]
            if not source_snapshot["use_relative"]:
                target_shape_keys.eval_time = source_snapshot["eval_time"]

            target_by_source_name = {source_basis_record["name"]: target_basis_key}
            writable_names = set()
            created_count = 0
            updated_count = 0

            for record in source_records[1:]:
                existing = target_shape_keys.key_blocks.get(record["name"])
                if existing is not None:
                    target_key = existing
                    target_by_source_name[record["name"]] = target_key
                    if settings.overwrite_existing or settings.clear_target:
                        writable_names.add(record["name"])
                        updated_count += 1
                else:
                    target_key = target.shape_key_add(name=record["name"], from_mix=False)
                    target_by_source_name[record["name"]] = target_key
                    writable_names.add(record["name"])
                    created_count += 1

            record_by_name = {record["name"]: record for record in source_records}
            use_relative = source_snapshot["use_relative"]

            for record in source_records[1:]:
                if record["name"] not in writable_names:
                    continue
                target_key = target_by_source_name[record["name"]]
                if use_relative:
                    target_key.relative_key = target_by_source_name.get(record["relative_name"], target_basis_key)
                _copy_key_metadata(target, record, target_key, settings.copy_values, use_relative)

            source_to_target = target.matrix_world.inverted_safe().to_3x3() @ source.matrix_world.to_3x3()
            processed = set()
            visiting = set()
            maximum_delta = {}

            window_manager.progress_begin(0, max(1, len(writable_names)))
            progress_started = True
            progress_index = 0

            def transfer_record(record):
                nonlocal progress_index
                name = record["name"]
                if name in processed or name not in writable_names:
                    return

                relative_name = record["relative_name"] if use_relative else source_basis_record["name"]
                if relative_name == name or relative_name not in record_by_name:
                    relative_name = source_basis_record["name"]

                if relative_name in visiting:
                    relative_name = source_basis_record["name"]
                elif relative_name in writable_names:
                    visiting.add(name)
                    transfer_record(record_by_name[relative_name])
                    visiting.discard(name)

                source_relative_record = record_by_name.get(relative_name, source_basis_record)
                target_key = target_by_source_name[name]
                target_relative_key = target_key.relative_key if use_relative else target_basis_key
                if target_relative_key is None:
                    target_relative_key = target_basis_key

                maximum_delta[name] = _write_transferred_key(
                    record,
                    source_relative_record,
                    target_key,
                    target_relative_key,
                    mapping,
                    mapping_method,
                    source_to_target,
                    target_count,
                )
                processed.add(name)
                progress_index += 1
                window_manager.progress_update(progress_index)

            for source_record in source_records[1:]:
                transfer_record(source_record)

            window_manager.progress_end()
            progress_started = False

            removed_count = 0
            if settings.remove_empty:
                threshold_squared = settings.empty_threshold * settings.empty_threshold
                protected_target_names = {
                    key_block.relative_key.name
                    for key_block in target_shape_keys.key_blocks[1:]
                    if key_block.relative_key is not None and key_block.relative_key != key_block
                }
                for record in reversed(source_records[1:]):
                    name = record["name"]
                    target_key = target_by_source_name.get(name)
                    if (
                        name in maximum_delta
                        and maximum_delta[name] <= threshold_squared
                        and target_key is not None
                        and target_key.name not in protected_target_names
                    ):
                        target.shape_key_remove(target_key)
                        removed_count += 1

            if not settings.clear_target and previous_active_name and target.data.shape_keys:
                previous_index = target.data.shape_keys.key_blocks.find(previous_active_name)
                target.active_shape_key_index = max(0, previous_index)
            else:
                target.active_shape_key_index = 0

            target.data.update()

            result_message = _tr("Transferred {count} shape keys ({removed} empty keys removed).").format(
                count=created_count + updated_count,
                removed=removed_count,
            )
            if missed_count:
                result_message += " " + _tr("{count} target vertices were outside the distance limit.").format(count=missed_count)
            self.report({"INFO"}, result_message)
            return {"FINISHED"}

        except Exception as error:  # Blender needs the full traceback in its console.
            if progress_started:
                window_manager.progress_end()
            traceback.print_exc()
            self.report({"ERROR"}, _tr("Shape key transfer failed: {error}").format(error=error))
            return {"CANCELLED"}


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
        settings = context.scene.topu_fst_settings
        source, target, message = _resolve_source_target(context)

        info_box = layout.box()
        if source is not None and target is not None:
            info_box.label(text=_tr("Source: {name}").format(name=source.name), icon="SHAPEKEY_DATA")
            info_box.label(text=_tr("Target (Active): {name}").format(name=target.name), icon="OBJECT_DATA")
        else:
            info_box.label(text=_tr(message), icon="INFO")
            info_box.label(text=_tr("Select the source first, then Shift-select the target."))

        layout.prop(settings, "mapping_method")

        distance_box = layout.box()
        distance_box.prop(settings, "use_max_distance")
        distance_row = distance_box.row()
        distance_row.enabled = settings.use_max_distance
        distance_row.prop(settings, "max_distance")

        layout.separator()
        clear_row = layout.row()
        clear_row.alert = settings.clear_target
        clear_row.prop(settings, "clear_target", icon="TRASH")

        overwrite_row = layout.row()
        overwrite_row.enabled = not settings.clear_target
        overwrite_row.prop(settings, "overwrite_existing")

        layout.prop(settings, "copy_values")
        layout.prop(settings, "remove_empty")
        threshold_row = layout.row()
        threshold_row.enabled = settings.remove_empty
        threshold_row.prop(settings, "empty_threshold")

        layout.separator()
        execute_row = layout.row()
        execute_row.scale_y = 1.4
        execute_row.enabled = source is not None and target is not None
        execute_row.operator(TOPU_OT_forced_shape_key_transfer.bl_idname, icon="MOD_DATA_TRANSFER")


def _draw_shape_key_menu(self, _context):
    self.layout.separator()
    self.layout.operator(TOPU_OT_forced_shape_key_transfer.bl_idname, icon="MOD_DATA_TRANSFER")


TRANSLATIONS = {
    "ja_JP": {
        ("*", "Forced Shape Key Transfer"): "シェイプキー強制転送",
        ("*", "Transfer all source shape keys to the active mesh even when topology differs"): "トポロジーが異なるメッシュ間でも、参照元の全シェイプキーをアクティブメッシュへ転送します",
        ("*", "Mapping"): "転送方式",
        ("*", "How source deformation is sampled for each target vertex"): "転送先の各頂点へ参照元の変形を割り当てる方式です",
        ("*", "Nearest Surface"): "最近傍面（推奨）",
        ("*", "Use the nearest source triangle and barycentric interpolation for smooth, accurate transfer"): "最も近い三角形から補間し、滑らかで精度の高い転送を行います",
        ("*", "Nearest Vertex (Legacy)"): "最近傍頂点（旧方式）",
        ("*", "Copy deformation from the single nearest source vertex, matching the original workflow"): "最も近い参照元頂点1つから変形をコピーする、元アドオン相当の方式です",
        ("*", "Remove Existing Target Shape Keys"): "転送先の既存シェイプキーを削除",
        ("*", "Remove every shape key on the target before transfer"): "転送前に転送先の全シェイプキーを削除します",
        ("*", "Overwrite Matching Shape Keys"): "同名シェイプキーを上書き",
        ("*", "Update target shape keys whose names match source shape keys"): "参照元と同名の転送先シェイプキーを更新します",
        ("*", "Remove Empty Shape Keys"): "変形のないシェイプキーを削除",
        ("*", "Remove transferred keys whose deformation is below the empty threshold, except keys used as relative bases"): "相対基準として使用されているキーを除き、変形量がしきい値以下の転送済みキーを削除します",
        ("*", "Empty Threshold"): "空判定しきい値",
        ("*", "Maximum deformation length considered empty in target local space"): "転送先ローカル空間で変形なしと判定する最大変形量です",
        ("*", "Limit Transfer Distance"): "転送距離を制限",
        ("*", "Leave target vertices unchanged when no source surface or vertex is within the specified world-space distance"): "指定したワールド空間距離内に参照元がない転送先頂点は変形させません",
        ("*", "Maximum Distance"): "最大距離",
        ("*", "Maximum world-space distance used for mapping"): "転送の対応付けに使用するワールド空間上の最大距離です",
        ("*", "Copy Current Values"): "現在値もコピー",
        ("*", "Copy each relative shape key's current value from the source"): "相対シェイプキーの現在値も参照元からコピーします",
        ("*", "Source: {name}"): "参照元: {name}",
        ("*", "Target (Active): {name}"): "転送先（アクティブ）: {name}",
        ("*", "Select the source first, then Shift-select the target."): "参照元を選び、続けてShiftで転送先をアクティブ選択してください。",
        ("*", "The active object must be a mesh."): "アクティブオブジェクトをメッシュにしてください。",
        ("*", "Switch to Object Mode before transferring."): "オブジェクトモードに切り替えてから実行してください。",
        ("*", "Select exactly two mesh objects."): "メッシュオブジェクトを2つだけ選択してください。",
        ("*", "The source and target must both be mesh objects."): "参照元と転送先は両方ともメッシュである必要があります。",
        ("*", "The source has no transferable shape keys."): "参照元に転送可能なシェイプキーがありません。",
        ("*", "The source and target meshes must contain vertices."): "参照元と転送先のメッシュには頂点が必要です。",
        ("*", "The source has no faces; nearest-vertex mapping was used instead."): "参照元に面がないため、最近傍頂点方式へ切り替えました。",
        ("*", "Transferred {count} shape keys ({removed} empty keys removed)."): "{count}個のシェイプキーを転送しました（空のキーを{removed}個削除）。",
        ("*", "{count} target vertices were outside the distance limit."): "転送先の{count}頂点が距離制限外でした。",
        ("*", "Shape key transfer failed: {error}"): "シェイプキー転送に失敗しました: {error}",
    }
}


CLASSES = (
    TOPU_FST_Settings,
    TOPU_OT_forced_shape_key_transfer,
    DATA_PT_topu_forced_shape_key_transfer,
)

_registered_menus = []


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.topu_fst_settings = PointerProperty(type=TOPU_FST_Settings)

    bpy.app.translations.register(ADDON_ID, TRANSLATIONS)

    for menu_name in ("MESH_MT_shape_key_context_menu", "MESH_MT_shape_key_specials"):
        menu_type = getattr(bpy.types, menu_name, None)
        if menu_type is not None:
            menu_type.append(_draw_shape_key_menu)
            _registered_menus.append(menu_type)


def unregister():
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

    if hasattr(bpy.types.Scene, "topu_fst_settings"):
        del bpy.types.Scene.topu_fst_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
