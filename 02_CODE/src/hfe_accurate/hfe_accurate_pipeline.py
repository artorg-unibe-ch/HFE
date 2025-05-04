"""
Script runs ACCURATE pipeline, converted from Denis's Bash script.

Author: Denis Schenk, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Maintained by: Simone Poncioni, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Date: April 2021
Latest update: 16.11.2023

UPDATES:
- Updated to run multiple simulations independently in parallel (POS)
"""

import logging
import os
import sys
from pathlib import Path
from shutil import move
from time import time

import hfe_abq.aim2fe as aim2fe
import hfe_abq.simulation as simulation
import hfe_accurate.postprocessing as postprocessing
import hfe_utils.imutils as imutils
import hfe_utils.print_optim_report as por
import numpy as np
import yaml
from hfe_utils.f_decomposition import decomposition_to_vtu
from hfe_utils.io_utils import print_mem_usage, write_timing_summary
from hfe_utils.odb2vtk_wrapper import Odb2VtkWrapper

os.environ["NUMEXPR_MAX_THREADS"] = "16"

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False

# flake8: noqa: E402, W503

"""
# TODO: reactivate this for mesh sensitivity analysis
# n_sim = int(15)  # has to match sweep in config
# min = 5, 5, 2, 7
# max = 20, 50, 10, 50 did not work, reducing to 20, 40, 10, 40
n_elms_longitudinal = np.linspace(1, 20, n_sim, dtype=int)
n_elms_transverse_trab = np.linspace(3, 50, n_sim, dtype=int)
n_elms_transverse_cort = np.linspace(1, 10, n_sim, dtype=int)
n_radial = np.linspace(3, 50, n_sim, dtype=int)

# update meshing settings with sweep factor for sensitivity analysis
sweep = cfg.meshing_settings.sweep_factor
cfg.meshing_settings.n_elms_longitudinal = int(
    n_elms_longitudinal[sweep - 1].item()
)
cfg.meshing_settings.n_elms_transverse_trab = int(
    n_elms_transverse_trab[sweep - 1].item()
)
cfg.meshing_settings.n_elms_transverse_cort = int(
    n_elms_transverse_cort[sweep - 1].item()
)
cfg.meshing_settings.n_elms_radial = int(n_radial[sweep - 1].item())
"""


def pipeline_hfe(cfg, folder_id, grayscale_filename):
    """
    Executes the homogenized finite elements (HFE) pipeline for a given sample.

    Args:
        cfg: Configuration object containing all necessary settings.
        folder_id (str): Identifier for the folder containing the sample data.
        grayscale_filename (str): Filename of the grayscale image to be processed.

    Returns:
        tuple: A tuple containing the time record dictionary and the summary path.
    """

    # timing
    time_record = {}
    start_full = time()
    start_sample = time()
    print_mem_usage()

    # Sets paths
    workdir = cfg.socket_paths.workdir
    feadir = Path(workdir, cfg.paths.feadir)
    umat = Path(workdir, cfg.abaqus.umat)
    sumdir = Path(workdir, cfg.paths.sumdir)
    sumdir.mkdir(parents=True, exist_ok=True)
    feadir.mkdir(parents=True, exist_ok=True)

    current_version = cfg.version.current_version

    sampledir = Path(feadir) / cfg.simulations.folder_id[grayscale_filename]

    inputfilename = f"{grayscale_filename}.inp".format(
        grayscale_filename, current_version
    )
    inputfile = sampledir / inputfilename
    sampledir.mkdir(parents=True, exist_ok=True)

    bone = {}
    bone, abq_inp_path = aim2fe.aim2fe_psl(cfg, grayscale_filename)

    if cfg.image_processing.BVTVd_comparison is True:
        bone = imutils.compute_bvtv_d_seg(bone, grayscale_filename)

    if cfg.mesher.meshing == "spline":
        inputfile = str(abq_inp_path.resolve())

    # 3.2) FZ_MAX
    cogs_dict = bone["cogs"]

    DIMZ = 0.0
    for arr in cogs_dict.values():
        for arr2 in arr.values():
            if arr2[-1] > DIMZ:
                DIMZ = arr2[-1]

    # create_loadcases.create_loadcase_fz_max(cfg, grayscale_filename, "FZ_MAX")

    start_simulation = time()
    try:
        odb_path = simulation.simulate_loadcase(
            cfg, grayscale_filename, inputfile, umat, ""
        )
        end_simulation = time()
    except Exception:
        logger.error("Simulation of FZ_MAX loadcase resulted in error")
        logger.error(sys.stderr)
        end_simulation = time()
        pass
    else:
        end_simulation = time()
    time_record["simulation"] = end_simulation - start_simulation

    optim = {}
    optim = postprocessing.datfilereader_psl(cfg, grayscale_filename, optim, "FZ_MAX")

    # timing
    end_sample = time()
    optim["processing_time"] = end_sample - start_sample
    time_record[grayscale_filename] = end_sample - start_sample

    path2dat = Path(inputfile).parent / (
        grayscale_filename + "_" + cfg.version.current_version[0:2] + ".dat"
    )
    thickness = max(val[2] for val in bone["nodes"].values()) - min(
        val[2] for val in bone["nodes"].values()
    )
    optim = por.compute_optim_report_variables(optim, path2dat, thickness)
    bone = por.compute_bone_report_variables_no_psl(bone)

    # only for sensitivity analysis
    mesh_parameters_dict = {
        "n_elms_longitudinal": cfg.meshing_settings.n_elms_longitudinal,
        "n_elms_transverse_trab": cfg.meshing_settings.n_elms_transverse_trab,
        "n_elms_transverse_cort": cfg.meshing_settings.n_elms_transverse_cort,
        "n_elms_radial": cfg.meshing_settings.n_elms_radial,
    }
    postprocessing.write_data_summary(
        cfg,
        optim,
        bone,
        grayscale_filename,
        mesh_parameters_dict,
        DOFs=bone["degrees_of_freedom"],
        time_sim=time_record[grayscale_filename],
    )

    if cfg.strain_localisation.strain_analysis is True:
        odb2vtkpath = cfg.socket_paths.odb2vtk
        odb_path = odb_path
        abq_path = cfg.solver.abaqus
        odb2vtk_wrapper = Odb2VtkWrapper(
            odb2vtkpath, odb_path, abq_path, only_last_frame=True
        )
        vtk_path = odb2vtk_wrapper.convert()
        print(f"ODB to VTK file written to {vtk_path}")
        decomposition_to_vtu(vtk_path)

    if cfg.abaqus.delete_odb is True:
        odbfilename = "{}_FZ_MAX_{}.odb".format(
            grayscale_filename, current_version[0:2]
        )
        odbfile = os.path.join(feadir, folder_id, odbfilename)
        os.remove(odbfile)

    sampledir_parent = Path(sampledir).parent

    try:
        # move whole content of feadir to sampledir except subdirectories
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                move(os.path.join(sampledir_parent, file), sampledir)
    except Exception:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        child_dir_time_path = (
            Path(sampledir) / f"simulation_current_time_{current_time}"
        )
        child_dir_time_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"File in this location already exists, moving to {child_dir_time_path}"
        )
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                move(os.path.join(sampledir_parent, file), child_dir_time_path)

    end_full = time()
    time_record["full"] = end_full - start_full

    print(yaml.dump(time_record, default_flow_style=False))

    write_timing_summary(cfg, grayscale_filename, time_record)
    return time_record
