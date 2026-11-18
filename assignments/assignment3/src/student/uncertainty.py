"""Statistical estimate of the registration error and the predicted TRE."""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation
from data_types.nominal_types import Frame
from ..mesh import Mesh
from .icp import IcpResult


def registration_covariance(result: IcpResult, F0_reg: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: implement.
    Left/CT convention: F*_reg = Exp(delta) . F_reg, eta_reg = [alpha; eps] relative to F0_reg.
    input: IcpResult, the initial guess F0_reg
    output: mu (6,) estimate of eta_reg, C_reg (6,6) covariance of delta
    """
    correction = result.F_reg * F0_reg.inv()
    mu = np.r_[Rotation.from_matrix(correction.R.matrix).as_rotvec(),
               correction.p.vec.ravel()]

    points = result.closest[result.inlier_mask]
    normals = result.normals[result.inlier_mask]
    distances = result.residuals[result.inlier_mask]
    dof = len(points) - 6
    if dof <= 0:
        raise ValueError("Registration covariance requires at least 7 inliers")

    # A surface match constrains motion along its normal to first order.
    J = np.column_stack((np.cross(points, normals), normals))
    _, singular_values, Vt = np.linalg.svd(J, full_matrices=False)
    cutoff = np.finfo(float).eps * max(J.shape) * singular_values[0]
    if singular_values[-1] <= cutoff:
        raise ValueError("Inlier surface geometry does not constrain all 6 pose components")

    sigma_squared = (distances @ distances) / dof
    C_reg = (Vt.T * (sigma_squared / singular_values**2)) @ Vt

    return mu, C_reg

def predicted_tre(mesh: Mesh, F_reg: Frame, C_reg: np.ndarray) -> dict:
    """
    TODO: implement.
    input: mesh, estimated F_reg, C_reg (6,6)
    output: {"rms", "mean", "max", "p90"} of the predicted TRE over the mesh vertices
    """
    x, y, z = mesh.vertices.T
    zeros = np.zeros_like(x)
    rotation_columns = np.stack((
        np.column_stack((zeros, z, -y)),
        np.column_stack((-z, zeros, x)),
        np.column_stack((y, -x, zeros)),
    ), axis=1)
    translation_columns = np.broadcast_to(np.eye(3), rotation_columns.shape)
    G = np.concatenate((rotation_columns, translation_columns), axis=2)
    squared_tre = np.einsum("vij,jk,vik->v", G, C_reg, G)
    squared_tre = np.maximum(squared_tre, 0.0)
    tre = np.sqrt(squared_tre)

    return {
        "rms": float(np.sqrt(squared_tre.mean())),
        "mean": float(tre.mean()),
        "max": float(tre.max()),
        "p90": float(np.percentile(tre, 90)),
    
    }
