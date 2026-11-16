"""Rigid registration: point-set registration, coarse initialisation and ICP."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from data_types.nominal_types import vct3, Frame, Rot
from ..mesh import Mesh, PatchBox
from .closest_point import closest_points_on_mesh



@dataclass
class IcpResult:
    """Output of icp()."""
    F_reg: Frame                  # reference body -> CT
    residuals: np.ndarray         # (K,) ||F_reg d_k - closest_k|| for every sample
    inlier_mask: np.ndarray       # (K,) bool, False for rejected samples
    closest: np.ndarray           # (K,3) closest mesh points of the last iteration
    normals: np.ndarray           # (K,3) mesh normals at those points
    n_iter: int
    converged: bool
    history: List[float] = field(default_factory=list)   # rms inlier residual per iteration

COLLINEAR_TOL = 1e-6

def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    input: paired points a_i, b_i (>= 3)
    output: least-squares Frame F with F * a_i ~= b_i
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same number of points")
    if len(a) < 3:
        raise ValueError("need at least 3 points")

    # turning the vct3 lists into numpy array -> each marker a,b becomes a row of A,B
    A = []
    for point in a:
        A.append(np.array(point.vec).reshape(3))
    B = []
    for point in b:
        B.append(np.array(point.vec).reshape(3))
    A = np.array(A)
    B = np.array(B)

    # center of each set
    a_mean = np.mean(A, axis=0)
    b_mean = np.mean(B, axis=0)

    # bring the center to the origin
    A_c = A - a_mean
    B_c = B - b_mean

    # collinear points check
    for centered in (A_c, B_c):
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[1] <= COLLINEAR_TOL * singular_values[0]:
            raise ValueError("points are collinear")

    # build H and get rotation using SVD
    H = np.zeros((3,3))
    for i in range(len(A_c)):
        H = H + np.outer(B_c[i], A_c[i])
    U, S, Vt = np.linalg.svd(H)     #h=USVt ->S is dropped so R=U*Vt
    R = U @ Vt

    # check if the rotation is a mirror image and fix
    if np.linalg.det(R) < 0:    # +1 = real rotation; -1 = mirror image
        U[:, 2] = -U[:, 2]
        R = U @ Vt

    # the translation -> b = Ra + p
    p = b_mean - R @ a_mean

    return Frame(Rot(matrix=R), vct3(p[0], p[1], p[2]))


def coarse_registration(d_by_patch: Dict[int, np.ndarray], boxes: List[PatchBox]) -> Frame:
    """
    TODO: implement.
    input: {patch id: (n_i,3) tip points in reference coordinates}, the patch boxes
    output: a rough F_reg (reference -> CT) from patch centroids <-> box centres
    """
    raise NotImplementedError("TODO: implement coarse_registration")


def icp(d: np.ndarray, mesh: Mesh, F_init: Frame, max_iter: int = 100, tol: float = 1e-4,
        reject: Optional[str] = "mad") -> IcpResult:
    """
    input: d (K,3) tip points in reference coordinates, mesh, initial F_reg,
           iteration cap, convergence tolerance, outlier-rejection method
    output: IcpResult
    """
    F_reg = F_init
    history = []
    converged = False
    
    prev_rms = np.inf

    for i in range(max_iter):
        # 1. move every d_k with the currect guess
        moved = []
        for k in d:
            moved_k = F_reg * vct3(k)
            moved.append(moved_k.vec.ravel())   
        moved = np.array(moved)

        # 2. closest mesh point for every moved point
        closest, tri_idx, dist  = closest_points_on_mesh(moved, mesh)

        # median absolute deviation (MAD)
        if reject == "mad":    
            m = np.median(dist)
            dist_from_m = abs(dist - m)
            mad = np.median(dist_from_m)
            limit = m + 3 * 1.4826 *  mad
            inlier_mask = dist <= limit
        else:
            inlier_mask = np.ones(len(d), dtype=bool)

        # inlier pairs
        d_list = []
        for row in d[inlier_mask]:
            d_list.append(vct3(row))

        closest_list = []
        for row in closest[inlier_mask]:
            closest_list.append(vct3(row))

        # 3. better frame from the pairs
        
        F_reg = register_points(d_list, closest_list)

        # 4. stop is rms barely changes
        curr_rms = np.sqrt(np.mean(np.square(dist[inlier_mask])))
        history.append(curr_rms)
        if abs(curr_rms - prev_rms) < tol:
            converged = True
            break
        prev_rms = curr_rms

    normals = mesh.face_normals[tri_idx]
    n_iter = i + 1

    return IcpResult(F_reg, dist, inlier_mask ,closest, normals, n_iter, converged, history)