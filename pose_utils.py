import numpy as np
import torch
import quaternion
import trimesh.transformations as tra


def apply_rotmat_to_poses(poses: torch.Tensor, rotm: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrices to pose vectors.
    poses: (J, 3) or (B, J, 3)
    rotm: (B, 3, 3)
    returns: rotated poses same shape as poses
    """
    B = rotm.shape[0]
    J = poses.shape[-2]
    flat_rot = rotm.view(B, 1, 3, 3).expand(B, J, 3, 3).reshape(B*J,3,3)
    flat_pose = poses.view(1, J, 3, 1).expand(B, J, 3, 1).reshape(B*J,3,1)
    out = torch.matmul(flat_rot, flat_pose).view(B, J, 3)
    return out


def normalize_vector(v: torch.Tensor, return_mag: bool=False):
    """
    Normalize last-dimension vectors to unit length.
    v: (..., 3)
    """
    mag = v.norm(dim=-1, keepdim=True)
    v_norm = v / mag.clamp(min=1e-8)
    if return_mag:
        return v_norm, mag.squeeze(-1)
    return v_norm


def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute cross product for (...,3) vectors.
    """
    return torch.stack([
        u[...,1]*v[...,2] - u[...,2]*v[...,1],
        u[...,2]*v[...,0] - u[...,0]*v[...,2],
        u[...,0]*v[...,1] - u[...,1]*v[...,0]
    ], dim=-1)


def rotation_matrix_from_ortho6d(ortho6d: torch.Tensor) -> torch.Tensor:
    """
    Construct rotation matrix from 6D continuous representation.
    ortho6d: (B,6)
    returns: (B,3,3)
    """
    x_raw = ortho6d[:,0:3]
    y_raw = ortho6d[:,3:6]
    x = normalize_vector(x_raw)
    z = normalize_vector(cross_product(x, y_raw))
    y = cross_product(z, x)
    R = torch.stack([x, y, z], dim=-1)
    return R


def pose_to_posquat(matrix: np.ndarray) -> tuple:
    """
    Convert 4x4 transform matrix to (position, quaternion) with quat as [x,y,z,w].
    """
    quat = tra.quaternion_from_matrix(matrix)
    pos = matrix[:3,3]
    return pos, np.array([quat[1], quat[2], quat[3], quat[0]])


def posquat_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Build 4x4 transform from position and quaternion [x,y,z,w].
    """
    q = [quat[3], quat[0], quat[1], quat[2]]
    T = tra.quaternion_matrix(q)
    T[:3,3] = pos
    return T


def l2_distance(x1: float, x2: float, y1: float, y2: float) -> float:
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def get_pose_from_transform(pos: np.ndarray, quat: np.ndarray) -> tuple:
    """
    Convert (pos,quat) to planar pose (x,y,theta) in radians.
    """
    q = quaternion.quaternion(quat[3], quat[0], quat[1], quat[2])
    euler = quaternion.as_euler_angles(q)
    theta = normalize_radians(euler[2])
    return pos[0], pos[1], theta


def relative_pose_change(p1: tuple, p2: tuple) -> tuple:
    """
    Given two poses (x,y,theta), compute relative dx,dy,dtheta.
    """
    x1,y1,o1 = p1
    x2,y2,o2 = p2
    dx = x2-x1
    dy = y2-y1
    dtheta = normalize_radians(o2 - o1)
    return dx, dy, dtheta


def update_pose(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Update batch poses by relative changes.
    pose: (B,3), delta: (B,3) with [dx,dy,dtheta(rad)]
    """
    x,y,theta = pose.unbind(-1)
    dx,dy,dth = delta.unbind(-1)
    new_x = x + dx*torch.cos(theta) - dy*torch.sin(theta)
    new_y = y + dx*torch.sin(theta) + dy*torch.cos(theta)
    new_theta = normalize_angle((theta + dth*180/np.pi))
    return torch.stack([new_x,new_y,new_theta], dim=-1)


def normalize_angle(angle_deg: float) -> float:
    """
    Normalize angle to [-180,180].
    """
    a = angle_deg % 360.0
    return a-360.0 if a>180.0 else a


def normalize_radians(angle_rad: float) -> float:
    """
    Normalize radians to [-pi,pi].
    """
    a = angle_rad % (2*np.pi)
    return a-2*np.pi if a>np.pi else a