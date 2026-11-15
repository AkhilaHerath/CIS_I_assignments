"""Closest-point queries against the triangle mesh."""

from typing import Tuple

import numpy as np

from data_types.nominal_types import vct3
from ..mesh import Mesh


def closest_point_on_triangle(p: vct3, a: vct3, b: vct3, c: vct3) -> vct3:
    """
    input: query point p, triangle corners a, b, c
    output: the point of the triangle (face, edge or vertex) closest to p
    """
    A = np.hstack([(b-a).vec, (c-a).vec])
    y = (p-a).vec
    x = np.linalg.inv(A.T @ A) @ (A.T @ y)
    lambda1 = x[0, 0]
    mu = x[1, 0]
    if lambda1 >= 0 and mu >= 0 and lambda1 + mu <= 1:
        return a + lambda1 * (b-a) + mu * (c-a)
    else:
        # check helper function (same shit)
        # A = (b-a).vec
        # x = np.linalg.inv(A.T@A) @ (A.T @ y)
        # lambda1 = x[0, 0]
        # if lambda1 >= 0 and lambda1 <= 1:
        #     return a + lambda1 * (b-a)
        # elif lambda1 < 0:
        #     return a
        # else:
        #     return b
        
        # check p against the three edges of the triangle
        q_ab = closest_points_on_segment(p, a, b)
        q_bc = closest_points_on_segment(p, b, c)
        q_ca = closest_points_on_segment(p, c, a)

        # pick the closest point among the three edges
        for q in [q_ab, q_bc, q_ca]:
            if p.distance(q) == min(p.distance(q_ab), p.distance(q_bc), p.distance(q_ca)):
                return q



def closest_points_on_mesh(points: np.ndarray, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TODO: implement (vectorised; a per-triangle Python loop is far too slow).
    input: points (K,3) in CT coordinates, mesh
    output: closest (K,3), triangle index (K,), distance (K,)
    """

    # points = np.asarray(points, dtype=float).reshape(-1, 3)

    closest = np.zeros((len(points), 3))
    tri_index = np.zeros(len(points), dtype=int)
    distances = np.zeros(len(points))

    for k in range(len(points)):
        p = points[k]
        pv = vct3(p)
        d = np.linalg.norm(p - mesh.centroids, axis=1)      # distance from point to triangle centroids
        r = mesh.bounding_sphere_radius                     # bounding sphere radius of each triangle

        # start with the triangle whose centroid is closest to the point
        closest_tri = np.argmin(d)                                    # triangle with the closest centroid
        verts = mesh.tri_vertices[closest_tri]                        # triangle vertices
        qv = closest_point_on_triangle(pv, vct3(verts[0]), vct3(verts[1]), vct3(verts[2]))        # closest point on triangle
        best = pv.distance(qv)                               # distance to beat
        
        # only check triangles whose bounding sphere intersects the best distance so far
        survivors = np.where(d - r <= best)[0]               # triangles that could have a closer point
        for tri in survivors:
            verts = mesh.tri_vertices[tri]              
            q = closest_point_on_triangle(pv, vct3(verts[0]), vct3(verts[1]), vct3(verts[2]))       
            dist = pv.distance(q)
            if dist < best:
                best = dist
                closest_tri = tri
                qv = q

        closest[k] = qv.vec.ravel()
        tri_index[k] = closest_tri
        distances[k] = best

    return closest, tri_index, distances





def closest_points_on_segment(p, start, end):
    """
    helper function to compute closest points on segments
    A: segment defined by start and end
    y: from start to p
    """
    A = (end-start).vec
    y = (p-start).vec
    x = np.linalg.inv(A.T@A) @ (A.T @ y)
    lambda1 = x[0, 0]
    if lambda1 >= 0 and lambda1 <= 1:
        return start + lambda1 * (end-start)
    elif lambda1 < 0:
        return start
    else:
        return end