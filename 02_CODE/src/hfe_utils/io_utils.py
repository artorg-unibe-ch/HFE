import logging
import os
from pathlib import Path
from socket import gethostname
from time import time

import numpy as np
import psutil
from omegaconf import OmegaConf

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def ext(filename, new_ext):
    """changes the file extension"""
    return filename.replace("." + filename.split(".")[-1], new_ext)


def print_mem_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_used_gb = mem_info.rss / (1024**3)  # Convert bytes to GB
    logger.debug(f"{'Memory usage:'.ljust(20)}\t\t{mem_used_gb:.2f} (GB)")
    return None


def timeit(method):
    def timed(*args, **kwargs):
        ts = time()
        result = method(*args, **kwargs)
        te = time() - ts
        # modify '30' if the function name is longer
        logger.debug(f"{method.__name__:<20}\t\t{te:.2f} (s)")
        print_mem_usage()
        return result

    return timed


def log_append_processingtime(filename, time):
    SUMname = filename
    time_summary = "\n".join(
        [
            "Summary Processing Time",
            "Full processing time           : {:.3f} [s]".format(time),
            "****************************************************************",
        ]
    )
    logger.info(time_summary)

    with open(SUMname, "a") as sumUpdate:
        sumUpdate.write("\n")
        sumUpdate.write(time_summary)
    logger.info("... added processing time to summary file")


def write_timing_summary(cfg, sample: str, time: dict):
    """
    Writes a summary of processing times for all samples in config
    to summaries folder.
    Parameters
    ----------
    path    path to store (summaries)
    time    dict with full processing time and simulation processing time

    Returns
    -------
    writes txt file
    """

    # file_path with pathlib from confing
    file_path = (
        Path(cfg.paths.sumdir)
        / f"{cfg.version.current_version}_processing_time_summary.csv"
    )

    with open(file_path, "w") as f:
        logger.debug(time, file=f)


def set_filenames(cfg, sample, pipeline="fast", origaim_separate=True):
    """
    Set filenames for each grayscale file.
    Filenames depend on pipeline (fast/accurate).
    Always:
    - native image for header
    - BMD or Native image croppped to ROI
    additional for fast pipeline:
    - periosteal mask
    additional for accurate pipeline:
    - trabecular mask
    - cortical mask
    - two-phase segmentation

    """

    # Always, not depending on nphase
    current_version = cfg.version.current_version
    folder_id = cfg.simulations.folder_id
    folder = folder_id[sample]
    feadir = Path(cfg.paths.feadir) / folder
    aimdir = Path(cfg.paths.aimdir) / folder
    origaimdir = Path(cfg.paths.origaimdir) / folder

    filename_postfix_bmd = cfg.filenames.filename_postfix_bmd

    filename = {}
    filename["FILEBMD"] = f"{sample}{filename_postfix_bmd}"
    if origaim_separate:
        filename["FILEGRAY"] = Path(sample).with_suffix(".AIM")
    else:
        filename["FILEGRAY"] = filename["FILEBMD"]
    filename["RAWname"] = str(Path(origaimdir) / filename["FILEGRAY"])
    filename["BMDname"] = str(Path(origaimdir) / filename["FILEBMD"])
    filename["boundary_conditions"] = cfg.paths.boundary_conditions

    # additional for fast pipeline
    if pipeline == "fast":
        filename_postfix_mask = cfg.filenames.filename_postfix_mask
        filename["FILEMASK"] = f"{sample}{filename_postfix_mask}"
        filename["MASKname"] = str(Path(origaimdir) / filename["FILEMASK"])

    # Additionl for accurate pipeline
    if pipeline == "accurate":
        filename_postfix_trab_mask = cfg.filenames.filename_postfix_trab_mask
        filename_postfix_cort_mask = cfg.filenames.filename_postfix_cort_mask
        filename_postfix_seg = cfg.filenames.filename_postfix_seg

        filename["FILEMASKCORT"] = f"{sample}{filename_postfix_cort_mask}"
        filename["FILEMASKTRAB"] = f"{sample}{filename_postfix_trab_mask}"
        filename["FILESEG"] = f"{sample}{filename_postfix_seg}"

        filename["CORTMASKname"] = str(Path(origaimdir) / filename["FILEMASKCORT"])
        filename["TRABMASKname"] = str(Path(origaimdir) / filename["FILEMASKTRAB"])
        filename["SEGname"] = str(Path(origaimdir) / filename["FILESEG"])

        if cfg.image_processing.mask_separate is False:
            filename_postfix_mask = cfg.filenames.filename_postfix_mask
            filename["FILEMASK"] = f"{sample}{filename_postfix_mask}"
            filename["MASKname"] = Path(origaimdir) / filename["FILEMASK"]

    # General filenames
    print(filename["BMDname"])
    new_filename = f"{sample}_V_{current_version}.inp"
    filename["INPname"] = str(Path(feadir) / new_filename)
    filename["VTKname"] = str(Path(aimdir) / new_filename)

    new_filename = f"{sample}_V_{current_version}_summary.txt"
    filename["SUMname"] = str(Path(feadir) / new_filename)

    new_filename = f"{sample}_V_{current_version}_BPVb"
    filename["VER_BPVname"] = str(Path(feadir) / new_filename)

    return filename


def hydra_update_cfg_key(cfg, key, value):
    # Make the config mutable
    OmegaConf.set_struct(cfg, False)

    # Split the key and access each level of the configuration separately
    keys = key.split(".")
    for k in keys[:-1]:
        cfg = cfg[k]

    setattr(cfg, keys[-1], value)

    # Make the config read-only again
    OmegaConf.set_struct(cfg, True)
    return None


def log_summary(bone, config, filenames, var):
    # bone
    # BMD_array = bone["BMD_array"]
    # Slope = bone["Slope"]
    # Intercept = bone["Intercept"]
    # Spacing = bone["Spacing"]
    # Scaling = bone["Scaling"]
    # elems = bone["elems"]
    # elems_bone = bone["elems_bone"]
    # FEelSize = bone["FEelSize"]
    # marray = bone["marray"]
    # mmarray1 = bone["mmarray1"]
    # mmarray2 = bone["mmarray2"]
    # mmarray3 = bone["mmarray3"]

    # # config
    # SCA1 = config["sca1"]
    # SCA2 = config["sca2"]
    # KMAX = config["kmax"]
    #
    # # PSL
    # mode_ghost_layer = config["mode_ghost_layer"]
    # n_ghost_proximal = config["padding_elements_proximal"]
    # n_ghost_distal = config["padding_elements_distal"]
    #
    # # filenames
    # BMDname = filenames["BMDname"]
    # SUMname = filenames["SUMname"]
    #
    # # Variables
    # mean_BMD = var["mean_BMD"]
    #
    #
    # BVTV_tissue = var["BVTV_tissue"]
    # mask_volume_MASK = var["Mask_Volume_MASK"]
    # mask_volume_FE = var["Mask_Volume_FE"]
    # mask_volume_quality = var["Volume_ratio"]
    # BV_tissue = var["BV_tissue"]
    # BVTV_FE_tissue_ROI = var["simulation_BVTV_FE_tissue_ROI"]
    # BVTV_FE_tissue_ELEM = var["simulation_BVTV_FE_tissue_ELEM"]
    #
    # BVTV_FE_elements_ROI = var["simulation_BVTV_FE_elements_ROI"]
    # BVTV_FE_elements_ELEM = var["simulation_BVTV_FE_elements_ELEM"]
    #
    # BVTV_quality_ROI = var["BVTV_ratio_ROI"]
    # BVTV_quality_ELEM = var["BVTV_ratio_ELEM"]
    #
    # BMC_tissue = var["BMC_tissue"]
    # BMC_FE_tissue_ROI = var["simulation_BMC_FE_tissue_ROI"]
    # BMC_FE_tissue_ELEM = var["simulation_BMC_FE_tissue_ELEM"]
    # BMC_quality_ROI = var["BMC_ratio_ROI"]
    # BMC_quality_ELEM = var["BMC_ratio_ELEM"]
    # # bone['']
    summary = "\n".join(
        [
            """
******************************************************************
**                         SUMMARY FILE                         **
**                hFE pipeline Denis Schenk 2018                **
******************************************************************""",
            "File                 : {}".format(filenames["BMDname"]),
            "System computed on   : {}".format(gethostname()),
            "Simulation Type      : Fast model (one phase / isotropic)",
            "Fitting Variables    : {:.2f} | {:.2f} | {:.3f} |".format(
                config["sca1"], config["sca2"], config["kmax"]
            ),
            "*****************************************************************",
            "Patient specific loading",
            "-------------------------------------------------------------------",
            "Ghost layer mode     : {}".format(config["mode_ghost_layer"]),
            "Ghost layers prox.   : {}".format(config["padding_elements_proximal"]),
            "Ghost layers dist.   : {}".format(config["padding_elements_distal"]),
            "*****************************************************************",
            "Image Dimension      : {}".format(np.shape(bone["BMD_array"])),
            "Scaling              : {}".format(bone["Scaling"]),
            "Slope                : {:.3f}".format(bone["Slope"]),
            "Intercept            : {:.3f}".format(bone["Intercept"]),
            "Spacing              : {:.3f}, {:.3f}, {:.3f} mm".format(*bone["Spacing"]),
            "FE element size      : {:.3f}, {:.3f}, {:.3f} mm".format(
                *bone["FEelSize"]
            ),
            "Number of elements   : {:d} + {:d}".format(
                len(bone["elems_bone"]), len(bone["elems"]) - len(bone["elems_bone"])
            ),
            "******************************************************************",
            "Variables computed from scaled BMD image and original masks",
            "-------------------------------------------------------------------",
            "*BMD*",
            "CORT                 : {:.2f} mgHA/ccm".format(var["CORT_mean_BMD_image"]),
            "TRAB                 : {:.2f} mgHA/ccm".format(var["TRAB_mean_BMD_image"]),
            "TOT                  : {:.2f} mgHA/ccm".format(var["TOT_mean_BMD_image"]),
            "*BMC*",
            "CORT                 : {:.2f} mgHA/ccm".format(var["CORT_mean_BMC_image"]),
            "TRAB                 : {:.2f} mgHA/ccm".format(var["TRAB_mean_BMC_image"]),
            "TOT                  : {:.2f} mgHA/ccm".format(var["TOT_mean_BMC_image"]),
            "*MASK Volumes*",
            "CORT                 : {:.2f} mm^3".format(var["Mask_Volume_CORTMASK"]),
            "TRAB                 : {:.2f} mm^3".format(var["Mask_Volume_TRABMASK"]),
            "TOT                  : {:.2f} mm^3".format(var["Mask_Volume_MASK"]),
            "******************************************************************",
            "Variables computed from elememnt values - original, without BMC conversion",
            "-------------------------------------------------------------------",
            "*BMC*",
            "CORT                 : {:.2f} mgHA/ccm".format(
                var["CORT_simulation_BMC_FE_tissue_orig_ROI"]
            ),
            "TRAB                 : {:.2f} mgHA/ccm".format(
                var["TRAB_simulation_BMC_FE_tissue_orig_ROI"]
            ),
            "TOT                  : {:.2f} mgHA/ccm".format(
                var["TOT_simulation_BMC_FE_tissue_orig_ROI"]
            ),
            "******************************************************************",
            "Variables computed from elememnt values - corrected, with BMC conversion",
            "-------------------------------------------------------------------",
            "*BMC*",
            "CORT                 : {:.2f} mgHA/ccm".format(
                var["CORT_simulation_BMC_FE_tissue_orig_ROI"]
            ),
            "TRAB                 : {:.2f} mgHA/ccm".format(
                var["TRAB_simulation_BMC_FE_tissue_ROI"]
            ),
            "TOT                  : {:.2f} mgHA/ccm".format(
                var["TOT_simulation_BMC_FE_tissue_ROI"]
            ),
            "******************************************************************",
            "Summary Benchmark Tests",
            "-------------------------------------------------------------------",
            "*without BMC conversion*",
            "BMC CORT             : {:.3f}".format(var["CORT_BMC_ratio_orig_ROI"]),
            "BMC TRAB             : {:.3f}".format(var["TRAB_BMC_ratio_orig_ROI"]),
            "BMC TOT              : {:.3f}".format(var["TOT_BMC_ratio_orig_ROI"]),
            "*with BMC conversion*",
            "BMC CORT             : {:.3f}".format(var["CORT_BMC_ratio_ROI"]),
            "BMC TRAB             : {:.3f}".format(var["TRAB_BMC_ratio_ROI"]),
            "BMC TOT              : {:.3f}".format(var["TOT_BMC_ratio_ROI"]),
            "*Volumes",
            "Mask volume CORT     : {:.3f}".format(var["CORTVolume_ratio"]),
            "Mask volume TRAB     : {:.3f}".format(var["TRABVolume_ratio"]),
            "Mask volume TOT      : {:.3f}".format(var["TOTVolume_ratio"]),
            "******************************************************************",
            "******************************************************************",
            # "Apparent variables computed from original BMD, BVTV images and mask",
            # "-------------------------------------------------------------------",
            # "*CORT*",
            # "Apparent BMD grays.  : {:.2f} mgHA/ccm".format(var['CORT_mean_BMD']),
            # "Apparent BVTV grays. : {:.3f} %".format(var['CORT_BVTV_tissue'] * 100.0),
            # "Apparent BMC grays.  : {:.1f} mgHA".format(var['CORT_BMC_tissue']),
            # "Apparent BV grays.   : {:.1f} mm^3".format(var['CORT_BV_tissue']),
            # "*TRAB*",
            # "Apparent BMD grays.  : {:.2f} mgHA/ccm".format(var['TRAB_mean_BMD']),
            # "Apparent BVTV grays. : {:.3f} %".format(var['TRAB_BVTV_tissue'] * 100.0),
            # "Apparent BMC grays.  : {:.1f} mgHA".format(var['TRAB_BMC_tissue']),
            # "Apparent BV grays.   : {:.1f} mm^3".format(var['TRAB_BV_tissue']),
            # "*TOT*",
            # "Apparent BMD grays.  : {:.2f} mgHA/ccm".format(var['TOT_mean_BMD']),
            # "Apparent BVTV grays. : {:.3f} %".format(var['TOT_BVTV_tissue'] * 100.0),
            # "Apparent BMC grays.  : {:.1f} mgHA".format(var['TOT_BMC_tissue']),
            # "Apparent BV grays.   : {:.1f} mm^3".format(var['TOT_BV_tissue']),
            # "******************************************************************",
            # "Volumes of mask",
            # "-------------------------------------------------------------------",
            # "*CORT*",
            # "Mask Volume image    : {:.1f} mm^3".format(var['Mask_Volume_CORTMASK']),
            # "Mask Volume FE mesh  : {:.1f} mm^3".format(var['CORTMask_Volume_FE']),
            # "Mask Volume ratio    : {:.3f} ".format(var['CORTVolume_ratio']),
            # "*TRAB*",
            # "Mask Volume image    : {:.1f} mm^3".format(var['Mask_Volume_TRABMASK']),
            # "Mask Volume FE mesh  : {:.1f} mm^3".format(var['TRABMask_Volume_FE']),
            # "Mask Volume ratio    : {:.3f} ".format(var['TRABVolume_ratio']),
            # "*TOT*",
            # "Mask Volume image    : {:.1f} mm^3".format(var['Mask_Volume_MASK']),
            # "Mask Volume FE mesh  : {:.1f} mm^3".format(var['CORTMask_Volume_FE']+var['TRABMask_Volume_FE']),
            # "Mask Volume ratio    : {:.3f} ".format(var['TOTVolume_ratio']),
            # "******************************************************************",
            # "BVTV",
            # "-------------------------------------------------------------------",
            # "*CORT*",
            # "BVTV tissue           : {:.2f} %".format(var['CORT_BVTV_tissue'] * 100.0),
            # "BVTV FE tissue ROI    : {:.2f} %".format(var['CORT_simulation_BVTV_FE_tissue_ROI'] * 100.0),
            # "BVTV FE tissue ELEM   : {:.2f} %".format(var['CORT_simulation_BVTV_FE_tissue_ELEM'] * 100.0),
            # "BVTV FE elements ROI  : {:.2f} %".format(var['CORT_simulation_BVTV_FE_elements_ROI'] * 100.0),
            # "BVTV FE elements ELEM : {:.2f} %".format(var['CORT_simulation_BVTV_FE_elements_ELEM'] * 100.0),
            # "BVTV ratio ROI        : {:.3f} ".format(var['CORT_BVTV_ratio_ROI']),
            # "BVTV ratio ELEM       : {:.3f} ".format(var['CORT_BVTV_ratio_ELEM']),
            # "*TRAB*",
            # "BVTV tissue           : {:.2f} %".format(var['TRAB_BVTV_tissue'] * 100.0),
            # "BVTV FE tissue ROI    : {:.2f} %".format(var['TRAB_simulation_BVTV_FE_tissue_ROI'] * 100.0),
            # "BVTV FE tissue ELEM   : {:.2f} %".format(var['TRAB_simulation_BVTV_FE_tissue_ELEM'] * 100.0),
            # "BVTV FE elements ROI  : {:.2f} %".format(var['TRAB_simulation_BVTV_FE_elements_ROI'] * 100.0),
            # "BVTV FE elements ELEM : {:.2f} %".format(var['TRAB_simulation_BVTV_FE_elements_ELEM'] * 100.0),
            # "BVTV ratio ROI        : {:.3f} ".format(var['TRAB_BVTV_ratio_ROI']),
            # "BVTV ratio ELEM       : {:.3f} ".format(var['TRAB_BVTV_ratio_ELEM']),
            # "******************************************************************",
            # "BMC",
            # "-------------------------------------------------------------------",
            # "*CORT*",
            # "BMC tissue            : {:.1f} mgHA".format(var['CORT_BMC_tissue']),
            # "BMC FE tissue ROI     : {:.1f} mgHA".format(var['CORT_simulation_BMC_FE_tissue_ROI']),
            # "BMC FE tissue ELEM    : {:.1f} mgHA".format(var['CORT_simulation_BMC_FE_tissue_ELEM']),
            # "BMC ratio ROI         : {:.3f} ".format(var['CORT_BMC_ratio_ROI']),
            # "BMC ratio ELEM        : {:.3f} ".format(var['CORT_BMC_ratio_ELEM']),
            # "*TRAB*",
            # "BMC tissue            : {:.1f} mgHA".format(var['TRAB_BMC_tissue']),
            # "BMC FE tissue ROI     : {:.1f} mgHA".format(var['TRAB_simulation_BMC_FE_tissue_ROI']),
            # "BMC FE tissue ELEM    : {:.1f} mgHA".format(var['TRAB_simulation_BMC_FE_tissue_ELEM']),
            # "BMC ratio ROI         : {:.3f} ".format(var['TRAB_BMC_ratio_ROI']),
            # "BMC ratio ELEM        : {:.3f} ".format(var['TRAB_BMC_ratio_ELEM']),
            # "*TOT*",
            # "BMC tissue            : {:.1f} mgHA".format(var['TOT_BMC_tissue']),
            # "BMC FE tissue ROI     : {:.1f} mgHA".format(var['TOT_simulation_BMC_FE_tissue_ROI']),
            # "BMC FE tissue ELEM    : {:.1f} mgHA".format(var['TOT_simulation_BMC_FE_tissue_ELEM']),
            # "BMC ratio ROI         : {:.3f} ".format(var['TOT_BMC_ratio_ROI']),
            # "BMC ratio ELEM        : {:.3f} ".format(var['TOT_BMC_ratio_ELEM']),
            # "******************************************************************",
            # "Average anisotropy",
            # "-------------------------------------------------------------------",
            # "Eigenvalues (max, mid, min)    : {0} {1} {2}".format(*marray),
            # "Degree of anisotropy (max/min) : {:.3f}".format(marray[0] / marray[2]),
            # "Eigenvector max (x,y,z)        : {}".format(mmarray1),
            # "Eigenvector mid (x,y,z)        : {}".format(mmarray2),
            # "Eigenvector min (x,y,z)        : {}".format(mmarray3),
            # "******************************************************************",
            # "Summary Benchmark Tests",
            # "-------------------------------------------------------------------",
            # "Benchmark tests compare variables computed from the full volume",
            # "against variables computed from the FE elements",
            # # "Volume Image/Subvolumes        : {:.3f}".format(mask_volume_quality),
            # # "BVTV ROI Image/Subvolumes      : {:.3f}".format(BVTV_quality_ROI),
            # # "BVTV ELEM Image/Subvolumes     : {:.3f}".format(BVTV_quality_ELEM),
            # # "BMC ROI Image/Subvolumes       : {:.3f}".format(BMC_quality_ROI),
            # # "BMC ELEM Image/Subvolumes      : {:.3f}".format(BMC_quality_ELEM),
            "******************************************************************",
        ]
    )

    print(summary)

    with open(filenames["SUMname"], "w") as sumFile:
        sumFile.write(summary)


class FileConfig:
    def __init__(self, cfg, sample: str, pipeline: str = "fast", origaim_separate: bool = True):
        self.cfg = cfg
        self.sample = sample
        self.pipeline = pipeline
        self.origaim_separate = origaim_separate
        self.current_version = cfg.version.current_version
        self.folder_id = cfg.simulations.folder_id
        self.folder = self.folder_id[sample]
        self.feadir = Path(cfg.paths.feadir) / self.folder
        self.aimdir = Path(cfg.paths.aimdir) / self.folder
        self.origaimdir = Path(cfg.paths.origaimdir) / self.folder
        self.filename_postfix_bmd = cfg.filenames.filename_postfix_bmd

    @property
    def file_bmd(self) -> Path:
        return Path(f"{self.sample}{self.filename_postfix_bmd}")

    @property
    def file_gray(self) -> Path:
        if self.origaim_separate:
            return Path(self.sample).with_suffix(".AIM")
        return self.file_bmd

    @property
    def raw_name(self) -> Path:
        return self.origaimdir / self.file_gray

    @property
    def bmd_name(self) -> Path:
        return self.origaimdir / self.file_bmd

    @property
    def boundary_conditions(self) -> str:
        return self.cfg.paths.boundary_conditions

    @property
    def file_mask(self) -> Path:
        if self.pipeline == "fast" or (self.pipeline == "accurate" and not self.cfg.image_processing.mask_separate):
            return Path(f"{self.sample}{self.cfg.filenames.filename_postfix_mask}")
        return None

    @property
    def mask_name(self) -> Path:
        if self.file_mask:
            return self.origaimdir / self.file_mask
        return None

    @property
    def file_mask_cort(self) -> Path:
        if self.pipeline == "accurate":
            return Path(f"{self.sample}{self.cfg.filenames.filename_postfix_cort_mask}")
        return None

    @property
    def file_mask_trab(self) -> Path:
        if self.pipeline == "accurate":
            return Path(f"{self.sample}{self.cfg.filenames.filename_postfix_trab_mask}")
        return None

    @property
    def file_seg(self) -> Path:
        if self.pipeline == "accurate":
            return Path(f"{self.sample}{self.cfg.filenames.filename_postfix_seg}")
        return None

    @property
    def cort_mask_name(self) -> Path:
        if self.file_mask_cort:
            return self.origaimdir / self.file_mask_cort
        return None

    @property
    def trab_mask_name(self) -> Path:
        if self.file_mask_trab:
            return self.origaimdir / self.file_mask_trab
        return None

    @property
    def seg_name(self) -> Path:
        if self.file_seg:
            return self.origaimdir / self.file_seg
        return None

    @property
    def inp_name(self) -> Path:
        return self.feadir / f"{self.sample}_V_{self.current_version}.inp"

    @property
    def vtk_name(self) -> Path:
        return self.aimdir / f"{self.sample}_V_{self.current_version}.inp"

    @property
    def sum_name(self) -> Path:
        return self.feadir / f"{self.sample}_V_{self.current_version}_summary.txt"

    @property
    def ver_bpv_name(self) -> Path:
        return self.feadir / f"{self.sample}_V_{self.current_version}_BPVb"

    def set_filenames(self):
        filename = {
            "FILEBMD": str(self.file_bmd),
            "FILEGRAY": str(self.file_gray),
            "RAWname": str(self.raw_name),
            "BMDname": str(self.bmd_name),
            "boundary_conditions": self.boundary_conditions,
            "INPname": str(self.inp_name),
            "VTKname": str(self.vtk_name),
            "SUMname": str(self.sum_name),
            "VER_BPVname": str(self.ver_bpv_name)
        }

        if self.pipeline == "fast":
            filename["FILEMASK"] = str(self.file_mask)
            filename["MASKname"] = str(self.mask_name)

        if self.pipeline == "accurate":
            filename["FILEMASKCORT"] = str(self.file_mask_cort)
            filename["FILEMASKTRAB"] = str(self.file_mask_trab)
            filename["FILESEG"] = str(self.file_seg)
            filename["CORTMASKname"] = str(self.cort_mask_name)
            filename["TRABMASKname"] = str(self.trab_mask_name)
            filename["SEGname"] = str(self.seg_name)

            if not self.cfg.image_processing.mask_separate:
                filename["FILEMASK"] = str(self.file_mask)
                filename["MASKname"] = str(self.mask_name)

        return filename
    
    # Usage
    # cfg = ...  # Your configuration object
    # sample = "sample_name"
    # file_config = FileConfig(cfg, sample)
    # print(file_config.bmd_name)
    # print(file_config.inp_name)