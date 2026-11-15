"""The end-to-end registration workflow: design, patch plan, data collection, registration."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..mesh import Mesh, PatchBox
from ..session import Session, SceneDesign, PointerPlacement
from .report import RegistrationReport
from .. import Frame, Rot, vct3


@dataclass
class SampleSet:
    """Tip positions collected through the tracker."""
    handles: List[PointerPlacement] = field(default_factory=list)
    patch_ids: List[int] = field(default_factory=list)
    d_ref: List[np.ndarray] = field(default_factory=list)   # (3,) tip in reference-body coordinates
    t_collect: float = 0.0

    def __len__(self) -> int:
        return len(self.d_ref)

    def as_array(self) -> np.ndarray:
        return np.array(self.d_ref).reshape(-1, 3)


def design_scene() -> SceneDesign:
    """
    implement -- return the same values the notebook's Step 1 cell uses
    (grading calls this instead of the notebook).
    """

    F_tracker_approx = Frame(Rot('x', np.pi), vct3(0, 0, 1900))     # Frame, camera pose relative to the workspace (looks along its local +z)
    ref_marker_positions = [    # >= 3 points, reference-body coordinates
        vct3(-45, -25, 0),
        vct3(50, -30, 0),
        vct3(0, 20, 0)
    ]
    ptr_marker_positions = [    # >= 3 points, pointer coordinates
        vct3(-50,  -20, 0),
        vct3( 50,  -20, 0),
        vct3(  0,   30, 0)
    ]
    p_tip = vct3(0, 0, -150)                # vct3, tip in pointer coordinates

    design = SceneDesign(F_tracker_approx, ref_marker_positions, ptr_marker_positions, p_tip)
    return design


def plan_patches(boxes: List[PatchBox], n_patches: int, samples_per_patch: int) -> List[int]:
    """
    input: the patch boxes, how many patches to use, samples per patch
    output: ordered list of patch ids to call place_pointer with (repeats allowed)
    """
    id_list = []
    for i in range(n_patches):
        for j in range(samples_per_patch):
            id_list.append(i)
    return id_list

        
    raise NotImplementedError("TODO: implement plan_patches")


def collect_samples(session: Session, plan: List[int]) -> SampleSet:
    """
    For every patch id: session.place_pointer, session.sample_marker_bodies,
    d_k = F_ref^-1 F_ptr p_tip; skip (and log) samples where a body was invisible.
    input: session, patch plan
    output: SampleSet
    """
    handles = []
    patch_ids = []
    d_ref = []
    t_start = session.elapsed()

    for patch_id in plan:
        handle = session.place_pointer(patch_id)
        obs = session.sample_marker_bodies()
        ref_obs = obs["ref"]
        pointer_obs = obs["ptr"]

        #skip and log samples where a body was invisible
        if ref_obs is None or pointer_obs is None:
            print(f"Sample for patch {patch_id} skipped due to missing observation.")
            continue

        F_ref = ref_obs.frame
        F_ptr = pointer_obs.frame

        d_k = F_ref.inv() * F_ptr * session.design.p_tip

        # save each sample to a list
        handles.append(handle)
        patch_ids.append(patch_id)
        d_ref.append(d_k.vec.ravel())
    
    t_collect = (session.elapsed() - t_start)
    
    return SampleSet(handles, patch_ids, d_ref, t_collect)


def run_registration(session: Session, mesh: Mesh, boxes: List[PatchBox], **kwargs) -> RegistrationReport:
    """
    TODO: implement -- the headless entry point used for grading:
    plan -> collect -> coarse_registration -> icp -> uncertainty -> RegistrationReport.
    input: session, mesh, boxes, your own keyword options
    output: RegistrationReport
    """
    raise NotImplementedError("TODO: implement run_registration")
