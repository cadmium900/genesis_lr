#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable, List, Optional, Sequence

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


def _get_fps(scene, fps_override: Optional[float]) -> float:
    if fps_override is not None:
        return float(fps_override)
    fbx = _require_fbx()
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    return float(fbx.FbxTime.GetFrameRate(time_mode))


def _quat_to_np(quat) -> np.ndarray:
    return np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)


def _quat_angle_rad(quat: np.ndarray) -> float:
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return 0.0
    quat = quat / norm
    w = abs(float(quat[3]))
    w = max(min(w, 1.0), -1.0)
    return 2.0 * math.acos(w)


def _quat_relative_angle_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    if np.linalg.norm(q1) < 1e-12 or np.linalg.norm(q2) < 1e-12:
        return 0.0
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = max(min(dot, 1.0), -1.0)
    return 2.0 * math.acos(dot)


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


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FBX animation bone rotations and print time/angle to console."
    )
    parser.add_argument("fbx_path", help="Path to FBX file.")
    parser.add_argument("--list", action="store_true", help="List skeleton bones and stacks.")
    parser.add_argument("--bones", default="", help="Comma-separated skeleton bone names to include.")
    parser.add_argument("--stacks", default="", help="Comma-separated animation stack names to include.")
    parser.add_argument("--fps", type=float, default=None, help="Override scene FPS.")
    parser.add_argument("--start-frame", type=int, default=None, help="Start frame index.")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step for sampling.")
    parser.add_argument("--delta", action="store_true", help="Report frame-to-frame angle delta.")
    parser.add_argument("--min-angle", type=float, default=0.0, help="Filter angles smaller than this.")
    unit = parser.add_mutually_exclusive_group()
    unit.add_argument("--degrees", action="store_true", help="Output angles in degrees (default).")
    unit.add_argument("--radians", action="store_true", help="Output angles in radians.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manager, scene = _load_scene(args.fbx_path)
    try:
        stacks = _get_anim_stacks(scene)
        bones = _collect_skeleton_nodes(scene)
        if args.list:
            print("# Skeleton bones")
            for node in bones:
                print(node.GetName())
            print("# Animation stacks")
            for stack in stacks:
                print(stack.GetName())
            return 0

        bones = _resolve_bones(bones, args.bones)
        stacks = _resolve_stacks(stacks, args.stacks)
        if not stacks:
            print("No animation stacks found.", file=sys.stderr)
            return 1

        fps = _get_fps(scene, args.fps)
        if fps <= 0:
            print("Invalid FPS.", file=sys.stderr)
            return 1

        use_degrees = not args.radians
        min_angle = float(args.min_angle)
        frame_step = max(1, int(args.frame_step))

        for stack in stacks:
            scene.SetCurrentAnimationStack(stack)
            span = stack.GetLocalTimeSpan()
            start_sec = span.GetStart().GetSecondDouble()
            end_sec = span.GetStop().GetSecondDouble()
            if args.start_frame is not None:
                start_sec = max(start_sec, args.start_frame / fps)
            if args.end_frame is not None:
                end_sec = min(end_sec, args.end_frame / fps)
            if end_sec < start_sec:
                print(f"Invalid frame range for stack {stack.GetName()}.", file=sys.stderr)
                continue

            total_frames = int(math.floor((end_sec - start_sec) * fps + 0.5)) + 1
            if total_frames <= 0:
                continue
            print(f"# Stack: {stack.GetName()} (frames={total_frames}, fps={fps:.3f})")

            prev_quats = [None] * len(bones)
            for frame_idx in range(0, total_frames, frame_step):
                time_sec = start_sec + frame_idx / fps
                if time_sec > end_sec + 1e-9:
                    break
                t = _require_fbx().FbxTime()
                t.SetSecondDouble(float(time_sec))
                for bone_idx, node in enumerate(bones):
                    local = node.EvaluateLocalTransform(t)
                    quat = _quat_to_np(local.GetQ())
                    if args.delta:
                        prev = prev_quats[bone_idx]
                        angle = _quat_relative_angle_rad(prev, quat) if prev is not None else 0.0
                    else:
                        angle = _quat_angle_rad(quat)
                    prev_quats[bone_idx] = quat
                    if use_degrees:
                        angle_out = math.degrees(angle)
                        label = "angle_deg"
                    else:
                        angle_out = angle
                        label = "angle_rad"
                    if angle_out < min_angle:
                        continue
                    print(
                        f"time={time_sec:.6f}s frame={frame_idx} bone={node.GetName()} "
                        f"{label}={angle_out:.6f}"
                    )
    finally:
        manager.Destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
