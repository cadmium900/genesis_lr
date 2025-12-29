from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _Rotation
except Exception:  # pragma: no cover - optional fallback
    _Rotation = None


class FbxImportError(RuntimeError):
    pass


def _require_fbx():
    try:
        import fbx  # type: ignore
    except Exception as exc:
        raise FbxImportError(
            "FBX SDK python bindings not found. Install Autodesk FBX SDK "
            "and ensure `import fbx` works."
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


def _load_fbx_scene(fbx_path: str):
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


def _collect_node_map(scene) -> Dict[str, object]:
    root = scene.GetRootNode()
    nodes = {}
    if root is None:
        return nodes
    for node in _iter_nodes(root):
        name = node.GetName()
        if name:
            nodes[name] = node
    return nodes


def _collect_skeleton_nodes(scene) -> List[str]:
    fbx = _require_fbx()
    nodes = _collect_node_map(scene)
    skeleton_names = []
    for name, node in nodes.items():
        attr = node.GetNodeAttribute()
        if attr is None:
            continue
        if _is_skeleton_attr(attr):
            skeleton_names.append(name)
    return skeleton_names


def _fbx_matrix_to_np(matrix) -> np.ndarray:
    out = np.zeros((4, 4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            out[r, c] = matrix.Get(r, c)
    return out


def _fbx_quat_to_np(quat) -> np.ndarray:
    return np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    if _Rotation is not None:
        return _Rotation.from_quat(quat).as_rotvec()
    q = quat / np.linalg.norm(quat)
    w = np.clip(q[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = q[:3] / s
    return axis * angle


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    if _Rotation is not None:
        return _Rotation.from_quat(quat).as_euler("xyz", degrees=False)
    x, y, z, w = quat
    # Standard XYZ intrinsic Euler extraction.
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _angle_from_quat(
    quat: np.ndarray,
    axis: Optional[np.ndarray],
    euler_axis: Optional[str],
) -> float:
    if axis is not None:
        axis = _normalize(axis)
        rotvec = _quat_to_rotvec(quat)
        return float(np.dot(rotvec, axis))
    if euler_axis is not None:
        euler = _quat_to_euler_xyz(quat)
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(euler_axis.lower())
        if axis_idx is None:
            raise ValueError(f"Invalid euler axis: {euler_axis}")
        return float(euler[axis_idx])
    raise ValueError("Need axis or euler axis to extract a dof angle.")


def _resolve_unit_scale(scene, unit_scale: Optional[float]) -> float:
    if unit_scale is not None:
        return float(unit_scale)
    unit = scene.GetGlobalSettings().GetSystemUnit()
    # FBX scale factor is relative to centimeters; convert to meters.
    return float(unit.GetScaleFactor() / 100.0)


def _resolve_fps(scene, fps: Optional[float]) -> float:
    if fps is not None:
        return float(fps)
    fbx = _require_fbx()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    return float(fbx.FbxTime.GetFrameRate(time_mode))


def list_fbx_skeleton_nodes(fbx_path: str) -> List[str]:
    manager, scene = _load_fbx_scene(fbx_path)
    try:
        return _collect_skeleton_nodes(scene)
    finally:
        manager.Destroy()


def list_fbx_animation_stacks(fbx_path: str) -> List[str]:
    fbx = _require_fbx()
    manager, scene = _load_fbx_scene(fbx_path)
    try:
        stacks = []
        criteria = _anim_stack_criteria()
        count = scene.GetSrcObjectCount(criteria)
        for idx in range(count):
            stack = scene.GetSrcObject(criteria, idx)
            stacks.append(stack.GetName())
        return stacks
    finally:
        manager.Destroy()


def load_fbx_mimic_data(
    fbx_path: str,
    dof_names: Sequence[str],
    dof_joint_map: Dict[str, str],
    effector_names: Sequence[str],
    dof_axis_map: Optional[Dict[str, Sequence[float]]] = None,
    dof_euler_map: Optional[Dict[str, str]] = None,
    dof_scale_map: Optional[Dict[str, float]] = None,
    dof_offset_map: Optional[Dict[str, float]] = None,
    root_name: Optional[str] = None,
    fps: Optional[float] = None,
    unit_scale: Optional[float] = None,
    axis_correction: Optional[np.ndarray] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load FBX animation stacks into mimic data.

    Returns a dict mapping animation stack names to:
      - dof_angles: (T, num_dofs)
      - dof_velocity: (T, num_dofs)
      - effector_pos: (T, num_effectors, 3)
      - dt: (T,)

    The `dof_axis_map` and `dof_euler_map` define how to extract single-DOF
    angles from each joint's local rotation. Provide at least one per DOF.
    """
    fbx = _require_fbx()
    manager, scene = _load_fbx_scene(fbx_path)
    try:
        nodes = _collect_node_map(scene)
        missing = []
        for dof_name in dof_names:
            joint_name = dof_joint_map.get(dof_name)
            if not joint_name or joint_name not in nodes:
                missing.append(dof_name)
        if missing:
            available = sorted(nodes.keys())
            raise KeyError(
                "Missing joints for DOFs: "
                + ", ".join(missing)
                + ". Available nodes include: "
                + ", ".join(available[:20])
            )

        effector_nodes = []
        for name in effector_names:
            if name not in nodes:
                raise KeyError(f"Missing effector node: {name}")
            effector_nodes.append(nodes[name])

        root_node = None
        if root_name is not None:
            if root_name not in nodes:
                raise KeyError(f"Missing root node: {root_name}")
            root_node = nodes[root_name]

        axis_correction = np.eye(3, dtype=np.float64) if axis_correction is None else axis_correction
        if axis_correction.shape != (3, 3):
            raise ValueError("axis_correction must be a 3x3 rotation matrix.")

        unit_scale = _resolve_unit_scale(scene, unit_scale)
        fps = _resolve_fps(scene, fps)

        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")

        dof_axis_map = dof_axis_map or {}
        dof_euler_map = dof_euler_map or {}
        dof_scale_map = dof_scale_map or {}
        dof_offset_map = dof_offset_map or {}

        stacks = {}
        criteria = _anim_stack_criteria()
        stack_count = scene.GetSrcObjectCount(criteria)
        for stack_idx in range(stack_count):
            stack = scene.GetSrcObject(criteria, stack_idx)
            scene.SetCurrentAnimationStack(stack)
            span = stack.GetLocalTimeSpan()
            start_sec = span.GetStart().GetSecondDouble()
            end_sec = span.GetStop().GetSecondDouble()
            if start_frame is not None:
                start_sec = max(start_sec, start_frame / fps)
            if end_frame is not None:
                end_sec = min(end_sec, end_frame / fps)
            if end_sec < start_sec:
                raise ValueError("Invalid frame range for animation stack.")
            frame_count = int(np.floor((end_sec - start_sec) * fps + 0.5)) + 1
            times_sec = start_sec + np.arange(frame_count, dtype=np.float64) / fps
            times_sec = times_sec[times_sec <= end_sec + 1e-9]
            if times_sec.size == 0:
                continue

            dt = np.diff(times_sec, prepend=times_sec[0]).astype(np.float32)
            dof_angles = np.zeros((times_sec.size, len(dof_names)), dtype=np.float32)
            effector_pos = np.zeros((times_sec.size, len(effector_nodes), 3), dtype=np.float32)

            for t_idx, sec in enumerate(times_sec):
                t = fbx.FbxTime()
                t.SetSecondDouble(float(sec))
                root_inv = None
                if root_node is not None:
                    root_mat = _fbx_matrix_to_np(root_node.EvaluateGlobalTransform(t))
                    root_inv = np.linalg.inv(root_mat)

                for dof_idx, dof_name in enumerate(dof_names):
                    joint_name = dof_joint_map[dof_name]
                    node = nodes[joint_name]
                    local_mat = node.EvaluateLocalTransform(t)
                    quat = _fbx_quat_to_np(local_mat.GetQ())
                    axis = dof_axis_map.get(dof_name)
                    if axis is not None:
                        axis = np.asarray(axis, dtype=np.float64)
                    angle = _angle_from_quat(quat, axis, dof_euler_map.get(dof_name))
                    scale = float(dof_scale_map.get(dof_name, 1.0))
                    offset = float(dof_offset_map.get(dof_name, 0.0))
                    dof_angles[t_idx, dof_idx] = angle * scale + offset

                for eff_idx, node in enumerate(effector_nodes):
                    eff_mat = _fbx_matrix_to_np(node.EvaluateGlobalTransform(t))
                    if root_inv is not None:
                        eff_mat = root_inv @ eff_mat
                    pos = eff_mat[:3, 3] * unit_scale
                    pos = axis_correction @ pos
                    effector_pos[t_idx, eff_idx] = pos.astype(np.float32)

            dof_velocity = np.zeros_like(dof_angles)
            if dof_angles.shape[0] > 1:
                for idx in range(1, dof_angles.shape[0]):
                    step = dt[idx] if dt[idx] > 0 else dt[idx - 1]
                    if step <= 0:
                        step = 1.0 / fps
                    dof_velocity[idx] = (dof_angles[idx] - dof_angles[idx - 1]) / step
                dof_velocity[0] = dof_velocity[1]

            stack_name = stack.GetName() or f"stack_{stack_idx}"
            stacks[stack_name] = {
                "dof_angles": dof_angles,
                "dof_velocity": dof_velocity,
                "effector_pos": effector_pos,
                "dt": dt,
            }

        return stacks
    finally:
        manager.Destroy()
