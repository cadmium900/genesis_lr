import numpy as np
import torch


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def vec_direction(a, b):
    """
    Unit direction vector pointing from a -> b.
    a, b: Tensors of shape (..., 3) representing points in 3D space.
    Returns:
        Tensor of shape (..., 3) representing the unit direction vectors.
    """
    # Compute the difference (b - a)
    diff = b - a

    # Compute the norm of the difference
    norm = torch.linalg.norm(diff, dim=-1, keepdim=True)

    # Avoid division by zero by setting zero norms to 1 (or handle as needed)
    norm = torch.where(norm == 0, torch.tensor(1.0, device=diff.device, dtype=diff.dtype), norm)

    # Compute the unit direction vector
    unit_direction = diff / norm

    return unit_direction

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_inv_quat(quat):
    qw, qx, qy, qz = quat.unbind(-1)
    inv_quat = torch.stack([1.0 * qw, -qx, -qy, -qz], dim=-1)
    return inv_quat

def angle_between_vectors(a, b, eps=1e-8):
    a_norm = normalize(a, eps)
    b_norm = normalize(b, eps)
    cos = (a_norm * b_norm).sum(dim=-1)                 # [N]
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)

def rotation_matrix_between_vectors(a, b, eps=1e-8):
    """
    Compute the rotation matrix that rotates vector `a` to vector `b`.

    Args:
        a (torch.Tensor): Tensor of shape (..., 3), the source vector(s).
        b (torch.Tensor): Tensor of shape (..., 3), the target vector(s).
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3).
    """
    # Normalize the input vectors
    a_norm = normalize(a, eps)
    b_norm = normalize(b, eps)

    # Compute the axis of rotation (cross product)
    axis = torch.cross(a_norm, b_norm, dim=-1)

    # Compute the angle of rotation (dot product)
    cos_theta = torch.sum(a_norm * b_norm, dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    # Compute the skew-symmetric cross-product matrix of the axis
    axis_norm = normalize(axis, eps)
    x, y, z = axis_norm.unbind(-1)
    skew_sym = torch.stack([
        torch.stack([torch.zeros_like(x), -z, y], dim=-1),
        torch.stack([z, torch.zeros_like(x), -x], dim=-1),
        torch.stack([-y, x, torch.zeros_like(x)], dim=-1),
    ], dim=-2)

    # Compute the rotation matrix using Rodrigues' rotation formula
    identity = torch.eye(3, device=a.device, dtype=a.dtype).expand_as(skew_sym)
    sin_theta = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
    rotation_matrix = identity + sin_theta * skew_sym + (1 - cos_theta) * torch.matmul(skew_sym, skew_sym)

    return rotation_matrix

def angle_on_plane(rotation_matrix, plane="XZ", eps=1e-8):
    """
    Extract the angle of rotation on a specified plane (e.g., XZ, XY, YZ) from a rotation matrix.

    Args:
        rotation_matrix (torch.Tensor): Tensor of shape (..., 3, 3), the rotation matrix.
        plane (str): The plane on which to compute the angle ("XZ", "XY", or "YZ").
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The angle on the specified plane in radians. Shape: (...).
    """
    # Define the plane indices
    if plane == "XZ":
        axis1, axis2 = 0, 2  # X and Z axes
    elif plane == "XY":
        axis1, axis2 = 0, 1  # X and Y axes
    elif plane == "YZ":
        axis1, axis2 = 1, 2  # Y and Z axes
    else:
        raise ValueError(f"Invalid plane '{plane}'. Choose from 'XZ', 'XY', or 'YZ'.")

    # Extract the relevant 2D rotation components from the rotation matrix
    vec1 = rotation_matrix[..., axis1, axis1]
    vec2 = rotation_matrix[..., axis1, axis2]

    # Compute the angle using atan2
    angle = torch.atan2(vec2, vec1)

    return angle

def quat_rotate(q, v):
    """
    Rotate vector(s) v by quaternion(s) q (wxyz order).
    q: [N,4], v: [N,3] -> returns [N,3]
    """
    q = normalize(q)
    w, x, y, z = q.unbind(-1)
    qv = torch.stack([x, y, z], dim=-1)                 # [N,3]
    t  = 2.0 * torch.cross(qv, v, dim=-1)               # [N,3]
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)

def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos

def quat_to_mat(quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    return rot_matrix

def gs_quat2euler(quat):  # xyz
    # Extract quaternion components
    qw, qx, qy, qz = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def gs_euler2quat(xyz):  # xyz

    roll, pitch, yaw = xyz.unbind(-1)

    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()

    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def gs_quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def gs_quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def gs_quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return gs_quat_apply(quat_yaw, vec)


def gs_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, :1], -a[:, 1:], ), dim=-1).view(shape)
