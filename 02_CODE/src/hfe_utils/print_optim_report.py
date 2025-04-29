from pathlib import Path

import numpy as np
from hfe_utils.read_mech_params import parse_and_calculate_stiffness_yield_force


def OR_ult_load_disp(optim: dict, loadcase: str) -> dict:
    """
    Computes maximum force and maximum moment and respective displacements and angles.

    Parameters
    ----------
    optim
    loadcase

    Returns
    -------
    optim dict
    """
    loadcase = str(loadcase)
    dict_loadcase_index = {
        "FX": 0,
        "FY": 1,
        "FZ": 2,
        "MX": 3,
        "MY": 4,
        "MZ": 5,
        "FZ_MAX": 2,
    }

    disp = np.asarray(optim["disp_" + loadcase])
    force_full = np.asarray(optim["force_" + loadcase])

    force = np.asarray(optim["force_" + loadcase])[:, 0:3]
    moment = np.asarray(optim["force_" + loadcase])[:, 3:6]

    # find index of maximum force values
    index_force = np.where(force == np.amax(force))

    # catch if several entries have same max value
    try:
        i_force_0 = int(index_force[0])
        max_force = force_full[i_force_0]
        disp_max_force = disp[i_force_0][dict_loadcase_index[loadcase]]
    except Exception:
        i_force_0 = int(index_force[0][0])
        max_force = force_full[i_force_0]
        disp_max_force = disp[i_force_0][dict_loadcase_index[loadcase]]

    # find index of maximum moment values
    index_moment = np.where(moment == np.amax(moment))

    # catch if several entries have same max value
    try:
        i_moment_0 = int(index_moment[0])
        max_moment = force_full[i_moment_0]
        disp_max_moment = disp[i_moment_0][dict_loadcase_index[loadcase]]
    except Exception:
        i_moment_0 = int(index_moment[0][0])
        max_moment = force_full[i_moment_0]
        disp_max_moment = disp[i_moment_0][dict_loadcase_index[loadcase]]

    optim["max_force_disp_" + loadcase] = [max_force, disp_max_force]
    optim["max_moment_disp_" + loadcase] = [max_moment, disp_max_moment]

    return optim


def compute_optim_report_variables(
    optim: dict, path2dat: Path, thickness_stacks: float
):

    optim = OR_ult_load_disp(optim, "FZ_MAX")

    force_FZ_MAX = np.array(optim["force_FZ_MAX"])
    disp_FZ_MAX = np.array(optim["disp_FZ_MAX"])

    stiffness = force_FZ_MAX[0][2] / disp_FZ_MAX[0][2]

    # Find maximum force
    for count, entry in enumerate(force_FZ_MAX[:, 2]):
        a = force_FZ_MAX[count, 2]
        try:
            b = force_FZ_MAX[count + 1, 2]
        except Exception:
            max_index = count
            break
        if a > b:
            max_index = count
            break

    # max_force = force_FZ_MAX[max_index, 2]
    # disp_at_max_force = disp_FZ_MAX[max_index, 2]

    stiffness, yield_force, yield_disp, max_force, disp_at_max_force = (
        parse_and_calculate_stiffness_yield_force(path2dat, thickness=thickness_stacks)
    )
    optim["yield_force_FZ_MAX"] = yield_force
    optim["yield_disp_FZ_MAX"] = yield_disp
    optim["stiffness_FZ_MAX"] = stiffness
    optim["max_force_FZ_MAX"] = max_force
    optim["disp_at_max_force_FZ_MAX"] = disp_at_max_force

    return optim


def compute_tissue_mineralization(
    bone: dict, SEG_array, BMD_array, string: str
) -> dict:
    """
    Compute tissue mineralization by masking BMD image with SEG and compute mean BMD of both phases

    Parameters
    ----------
    bone
    SEG_array
    BMD_array
    string

    Returns
    -------
    bone

    """
    cortmask = bone["CORTMASK_array"]
    trabmask = bone["TRABMASK_array"]

    cortmask[cortmask > 0.0] = 1.0
    trabmask[trabmask > 0.0] = 1.0
    SEG_array[SEG_array > 0.0] = 1.0

    BMD_array_cort = BMD_array * cortmask
    BMD_array_trab = BMD_array * trabmask
    bone["mean_BMD_SEG_CORT" + string] = np.nanmean(BMD_array_cort[SEG_array == 1])
    bone["mean_BMD_SEG_TRAB" + string] = np.nanmean(BMD_array_trab[SEG_array == 1])

    return bone


def compute_bone_volume(bone: dict, SEG_array) -> dict:
    """
    Computes bone volume from segmentation SEG_array

    Parameters
    ----------
    bone
    SEG_array

    Returns
    -------
    """

    voxel_volume = bone["Spacing"][0] ** 3

    cortmask = bone["CORTMASK_array"]
    trabmask = bone["TRABMASK_array"]
    SEG_array_cort = SEG_array * cortmask
    SEG_array_trab = SEG_array * trabmask

    n_voxel_CORT = np.count_nonzero(SEG_array_cort[SEG_array_cort == 1])
    n_voxel_TRAB = np.count_nonzero(SEG_array_trab[SEG_array_trab == 1])

    bone["BV_CORT_SEG"] = n_voxel_CORT * voxel_volume
    bone["BV_TRAB_SEG"] = n_voxel_TRAB * voxel_volume

    return bone


def compute_bone_report_variables_no_psl(bone: dict) -> dict:

    bone = compute_tissue_mineralization(
        bone, bone["SEG_array"], bone["BMD_array"], "orig"
    )
    bone = compute_tissue_mineralization(
        bone, bone["SEG_array"], bone["BMDscaled"], "scaled"
    )

    bone = compute_bone_volume(bone, bone["SEG_array"])

    return bone
