#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

def _require_fbx():
    try:
        import fbx  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FBX SDK python bindings not found. Ensure the Autodesk FBX SDK "
            "wheel is installed for this Python interpreter."
        ) from exc
    return fbx


def _anim_stack_criteria():
    fbx = _require_fbx()
    return fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId)


def _is_skeleton_attr(attr) -> bool:
    fbx = _require_fbx()
    attr_type = attr.GetAttributeType()
    if hasattr(fbx.FbxNodeAttribute, "eSkeleton"):
        return attr_type == fbx.FbxNodeAttribute.eSkeleton
    return attr_type == fbx.FbxNodeAttribute.EType.eSkeleton


def _load_scene(fbx_path: str):
    fbx = _require_fbx()
    try:
        import FbxCommon  # type: ignore
    except Exception:
        FbxCommon = None

    if FbxCommon is not None and hasattr(FbxCommon, "InitializeSdkObjects"):
        manager, scene = FbxCommon.InitializeSdkObjects()
        if not FbxCommon.LoadScene(manager, scene, fbx_path):
            raise RuntimeError(f"Failed to load FBX scene: {fbx_path}")
        return manager, scene

    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(fbx_path, -1, manager.GetIOSettings()):
        status = importer.GetStatus()
        raise RuntimeError(f"Failed to load FBX scene: {fbx_path}: {status.GetErrorString()}")
    scene = fbx.FbxScene.Create(manager, "scene")
    importer.Import(scene)
    importer.Destroy()
    return manager, scene


def _iter_nodes(root) -> Iterable:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for idx in range(node.GetChildCount()):
            stack.append(node.GetChild(idx))


def _collect_skeleton_nodes(scene) -> List:
    root = scene.GetRootNode()
    nodes = []
    if root is None:
        return nodes
    for node in _iter_nodes(root):
        attr = node.GetNodeAttribute()
        if attr is None:
            continue
        if _is_skeleton_attr(attr):
            nodes.append(node)
    return nodes


def _get_anim_stacks(scene) -> List:
    stacks = []
    criteria = _anim_stack_criteria()
    count = scene.GetSrcObjectCount(criteria)
    for idx in range(count):
        stacks.append(scene.GetSrcObject(criteria, idx))
    return stacks


def _find_calibration_stack(stacks: Sequence) -> Optional:
    for stack in stacks:
        if "robot|calibration" in stack.GetName().lower():
            return stack
    return None


def _get_fps(scene, fps_override: Optional[float]) -> float:
    if fps_override is not None:
        return float(fps_override)
    fbx = _require_fbx()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    return float(fbx.FbxTime.GetFrameRate(time_mode))


def _get_scene_unit_scale(scene) -> float:
    fbx = _require_fbx()
    system_unit = scene.GetGlobalSettings().GetSystemUnit()
    # FBX scale factor is in centimeters per unit.
    scale_cm = float(system_unit.GetScaleFactor())
    return scale_cm * 0.01


def _quat_to_np(quat) -> np.ndarray:
    return np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_inv(quat: np.ndarray) -> np.ndarray:
    norm = float(np.dot(quat, quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64) / norm


def _euler_deg_to_quat_xyz(euler_deg: Sequence[float]) -> np.ndarray:
    fbx = _require_fbx()
    mat = fbx.FbxAMatrix()
    mat.SetR(fbx.FbxVector4(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])))
    return _quat_to_np(mat.GetQ())


def _euler_rad_to_quat_xyz(euler_rad: Sequence[float]) -> np.ndarray:
    euler_deg = np.degrees(np.asarray(euler_rad, dtype=np.float64))
    return _euler_deg_to_quat_xyz(euler_deg)


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _local_quat_from_global(node, t) -> np.ndarray:
    global_quat = _quat_to_np(node.EvaluateGlobalTransform(t).GetQ())
    parent = node.GetParent()
    if parent is None:
        return global_quat
    parent_quat = _quat_to_np(parent.EvaluateGlobalTransform(t).GetQ())
    return _quat_mul(_quat_inv(parent_quat), global_quat)


def _global_pos_from_node(node, t) -> np.ndarray:
    pos = node.EvaluateGlobalTransform(t).GetT()
    return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)


def _unwrap_angle(prev: Optional[float], current: float) -> float:
    if prev is None:
        return current
    while current - prev > math.pi:
        current -= 2.0 * math.pi
    while current - prev < -math.pi:
        current += 2.0 * math.pi
    return current


def _unwrap_angles(angles: Sequence[float]) -> List[float]:
    if not angles:
        return []
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        unwrapped.append(_unwrap_angle(unwrapped[-1], angle))
    return unwrapped


def _compute_urdf_joint_angles(
    bones: Sequence,
    t,
    calibration_map: Dict[str, Tuple[int, float, float, float, float, str, np.ndarray, np.ndarray]],
    rest_quats: Optional[Dict[str, np.ndarray]] = None,
    prev_axis_angles: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
    if prev_axis_angles is None:
        prev_axis_angles = {}
    urdf_angles: Dict[str, float] = {}
    axis_angles: Dict[str, float] = {}
    local_quats: Dict[str, np.ndarray] = {}
    for node in bones:
        bone_name = node.GetName()
        calib = calibration_map.get(bone_name)
        if calib is None:
            continue
        direction, offset, _, _, _, _, axis_fbx, ref_quat = calib
        local_quat = _local_quat_from_global(node, t)
        local_quats[bone_name] = local_quat
        quat = local_quat
        if rest_quats is not None:
            rest_quat = rest_quats.get(bone_name)
            if rest_quat is not None:
                quat = _quat_mul(_quat_inv(rest_quat), quat)
        delta_quat = _quat_mul(_quat_inv(ref_quat), quat)
        axis_angle_raw = _axis_angle_rad(delta_quat, axis_fbx)
        prev_angle = prev_axis_angles.get(bone_name)
        axis_angle = _unwrap_angle(prev_angle, axis_angle_raw)
        prev_axis_angles[bone_name] = axis_angle
        axis_angles[bone_name] = axis_angle
        urdf_angles[bone_name] = direction * axis_angle + offset
    return urdf_angles, axis_angles, local_quats


def _quat_angle_rad(quat: np.ndarray) -> float:
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return 0.0
    quat = quat / norm
    w = abs(float(quat[3]))
    w = max(min(w, 1.0), -1.0)
    return 2.0 * math.acos(w)


def _quat_axis_angle(quat: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    quat = quat / norm
    if quat[3] < 0:
        quat = -quat
    w = max(min(float(quat[3]), 1.0), -1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = quat[:3] / s
    return axis, float(angle)


def _deduce_axis_from_quats(quats: Sequence[np.ndarray]) -> np.ndarray:
    axis_accum = np.zeros(3, dtype=np.float64)
    for quat in quats:
        axis, angle = _quat_axis_angle(quat)
        if angle < 1e-4:
            continue
        if np.linalg.norm(axis_accum) > 1e-6 and np.dot(axis_accum, axis) < 0:
            axis = -axis
        axis_accum += axis * angle
    norm = np.linalg.norm(axis_accum)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis_accum / norm


def _axis_angle_rad(quat: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return 0.0
    axis = axis / axis_norm
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    if quat[3] < 0:
        quat = -quat
    w = max(min(float(quat[3]), 1.0), -1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        rot_axis = quat[:3] / s
    return float(np.dot(rot_axis, axis) * angle)


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _solve_calibration_mapping(
    fbx_min: float,
    fbx_max: float,
    lower: float,
    upper: float,
    tol: float = 1e-3,
) -> Tuple[int, float, float, str]:
    if fbx_max < fbx_min:
        fbx_min, fbx_max = fbx_max, fbx_min
    if abs(fbx_max - fbx_min) < 1e-6:
        return 1, 0.0, float("inf"), "range_too_small"
    offset_pos = lower - fbx_min
    err_pos = abs((fbx_max + offset_pos) - upper)
    offset_neg = lower + fbx_max
    err_neg = abs((-fbx_min + offset_neg) - upper)
    if err_pos <= err_neg:
        direction = 1
        offset = offset_pos
        err = err_pos
    else:
        direction = -1
        offset = offset_neg
        err = err_neg
    status = "ok" if err <= tol else "mismatch"
    return direction, float(offset), float(err), status


def _build_calibration_map(
    scene,
    stack,
    bones: Sequence,
    bone_limit_map: Dict[str, Tuple[np.ndarray, float, float]],
    rest_quats: Optional[Dict[str, np.ndarray]],
    fps: float,
    end_frame: int = 50,
) -> Dict[str, Tuple[int, float, float, float, float, str, np.ndarray, np.ndarray]]:
    fbx = _require_fbx()
    scene.SetCurrentAnimationStack(stack)
    span = stack.GetLocalTimeSpan()
    start_sec = max(span.GetStart().GetSecondDouble(), 0.0)
    end_sec = min(span.GetStop().GetSecondDouble(), end_frame / fps)
    if end_sec < start_sec:
        return {}
    total_frames = int(math.floor((end_sec - start_sec) * fps + 0.5)) + 1
    if total_frames <= 0:
        return {}
    cal_quats: Dict[str, List[np.ndarray]] = {name: [] for name in bone_limit_map.keys()}
    ref_quats: Dict[str, np.ndarray] = {}
    t_ref = fbx.FbxTime()
    t_ref.SetSecondDouble(float(start_sec))
    for node in bones:
        bone_name = node.GetName()
        if bone_name not in cal_quats:
            continue
        local_quat = _local_quat_from_global(node, t_ref)
        if rest_quats is not None:
            rest_quat = rest_quats.get(bone_name)
            if rest_quat is not None:
                local_quat = _quat_mul(_quat_inv(rest_quat), local_quat)
        ref_quats[bone_name] = local_quat
    for frame_idx in range(total_frames):
        time_sec = start_sec + frame_idx / fps
        if time_sec > end_sec + 1e-9:
            break
        t = fbx.FbxTime()
        t.SetSecondDouble(float(time_sec))
        for node in bones:
            bone_name = node.GetName()
            if bone_name not in cal_quats:
                continue
            ref_quat = ref_quats.get(bone_name)
            if ref_quat is None:
                continue
            quat = _local_quat_from_global(node, t)
            if rest_quats is not None:
                rest_quat = rest_quats.get(bone_name)
                if rest_quat is not None:
                    quat = _quat_mul(_quat_inv(rest_quat), quat)
            delta_quat = _quat_mul(_quat_inv(ref_quat), quat)
            cal_quats[bone_name].append(delta_quat)
    calibration_map: Dict[str, Tuple[int, float, float, float, float, str, np.ndarray, np.ndarray]] = {}
    for bone_name, quats in cal_quats.items():
        if not quats:
            continue
        axis_fbx = _deduce_axis_from_quats(quats)
        angles = [_axis_angle_rad(quat, axis_fbx) for quat in quats]
        angles = _unwrap_angles(angles)
        fbx_min = float(min(angles))
        fbx_max = float(max(angles))
        _, lower, upper = bone_limit_map[bone_name]
        direction, offset, err, status = _solve_calibration_mapping(
            fbx_min, fbx_max, lower, upper
        )
        #print(f"{bone_name} CAL fbx_min {fbx_min}  fbx_max {fbx_max}   lower {lower}  upper {upper}")
        ref_quat = ref_quats.get(bone_name)
        if ref_quat is None:
            continue
        calibration_map[bone_name] = (
            direction,
            offset,
            fbx_min,
            fbx_max,
            err,
            status,
            axis_fbx,
            ref_quat,
        )
    return calibration_map


def _parse_urdf_joint_info(
    urdf_path: str,
) -> Dict[str, Tuple[np.ndarray, float, float, np.ndarray]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: Dict[str, Tuple[np.ndarray, float, float, np.ndarray]] = {}
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype not in ("revolute", "continuous"):
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            continue
        axis_elem = joint.find("axis")
        if axis_elem is None or not axis_elem.get("xyz"):
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = np.array([float(x) for x in axis_elem.get("xyz").split()], dtype=np.float64)
        origin = joint.find("origin")
        if origin is None or not origin.get("rpy"):
            rpy = np.zeros(3, dtype=np.float64)
        else:
            rpy = np.array([float(x) for x in origin.get("rpy").split()], dtype=np.float64)
        child = joint.find("child")
        child_link = child.get("link") if child is not None else ""
        name = joint.get("name") or ""
        key = child_link or name
        joints[key] = (axis, float(lower), float(upper), rpy)
        if name:
            joints[name] = (axis, float(lower), float(upper), rpy)
    return joints


def _resolve_bones(all_nodes: Sequence, bone_names: Optional[str]) -> List:
    if not bone_names:
        return list(all_nodes)
    requested = [name.strip() for name in bone_names.split(",") if name.strip()]
    nodes = []
    name_map = {node.GetName(): node for node in all_nodes}
    missing = []
    for name in requested:
        node = name_map.get(name)
        if node is None:
            missing.append(name)
        else:
            nodes.append(node)
    if missing:
        available = ", ".join(sorted(name_map.keys())[:20])
        raise RuntimeError(f"Missing bones: {', '.join(missing)}. Available: {available}")
    return nodes


def _resolve_stacks(all_stacks: Sequence, stack_names: Optional[str]) -> List:
    if not stack_names:
        return list(all_stacks)
    requested = [name.strip() for name in stack_names.split(",") if name.strip()]
    stacks = []
    name_map = {stack.GetName(): stack for stack in all_stacks}
    missing = []
    for name in requested:
        stack = name_map.get(name)
        if stack is None:
            missing.append(name)
        else:
            stacks.append(stack)
    if missing:
        available = ", ".join(sorted(name_map.keys())[:20])
        raise RuntimeError(f"Missing stacks: {', '.join(missing)}. Available: {available}")
    return stacks
