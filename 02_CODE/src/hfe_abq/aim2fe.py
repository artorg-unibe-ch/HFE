"""
Converts an AIM file to a HDF5 file.
Adapted from the script from Denis Schenk, ISTB, University of Bern.

Author: Jarunan Panyasantisuk, ETH Scientific IT Services.
Date: 13 November 2018.

Maintained by: Simone Poncioni, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Date: March 2024
"""

import gc
import logging
import threading
from pathlib import Path
from time import sleep

import hfe_accurate.material_mapping as material_mapping
import hfe_accurate.preprocessing as preprocessing
import hfe_utils.imutils as imutils
import hfe_utils.io_utils as io_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from pyhexspline.spline_mesher import HexMesh  # type: ignore

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass


def save_image_with_colorbar(data, output_path):
    """
    Saves an image with a colorbar to the specified output path.

    Args:
        data (numpy.ndarray): The image data to be displayed.
        output_path (str): The path where the image will be saved.

    Returns:
        None

    This function performs the following operations:
    1. Creates a new figure and axis.
    2. Displays the image data using a viridis colormap.
    3. Sets the title of the image based on the output path.
    4. Adds a colorbar to the right of the image.
    5. Adjusts the layout to be tight.
    6. Saves the figure to the specified output path with a resolution of 300 dpi.
    7. Clears the figure to free up memory.
    """

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data, cmap="viridis")
    plt.title(
        f"{Path(output_path).parent.name} - {Path(output_path).stem}",
        fontsize=20,
        weight="bold",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.clf()


def aim2fe_psl(cfg, sample):
    """
    Converts AIM image to Abaqus INP file using the provided configuration.

    Args:
        cfg (dict): Dictionary of configuration parameters.
        sample (str): Sample identifier.

    Raises:
        TypeError: If there is an issue with the input types.
        ValueError: If an unrecognized fabric or meshing type is encountered.

    Returns:
        tuple: A tuple containing the bone dictionary and the path to the Abaqus INP file.

    This function performs the following operations:
    1. Sets filenames and reads image parameters.
    2. Reads AIM images and image parameters, optionally using multithreading.
    3. Adjusts image size if registration is enabled.
    4. Saves images with colorbars.
    5. Prepares material mapping by calculating BVTV and BMD values.
    6. Generates a mesh if the meshing type is "spline".
    7. Computes MSL kernel list based on the fabric type.
    8. Maps materials to the mesh.
    9. Computes and stores summary and performance variables.
    10. Plots MSL fabric if the meshing type is "spline".
    """

    # For Hosseini Dataset, Image parameters are read from original aim, not from processed BMD file, as there
    # they were deleted by medtool pre-processing
    origaim_separate_bool = cfg.image_processing.origaim_separate

    # filenames = io_utils.set_filenames(
    #     cfg, sample, pipeline="accurate", origaim_separate=origaim_separate_bool
    # )

    filenames = io_utils.FileConfig(cfg, sample, pipeline='accurate', origaim_separate=origaim_separate_bool)
    filenames.set_filenames()
    print(yaml.dump(filenames, default_flow_style=False))
    io_utils.print_mem_usage()

    # 2 Read AIM images and image parameters
    # ---------------------------------------------------------------------------------
    bone = {}
    bone["sample"] = str(sample)

    spacing, scaling, slope, intercept = imutils.read_img_param(filenames)
    bone["Spacing"] = spacing
    bone["Scaling"] = scaling
    bone["Slope"] = slope
    bone["Intercept"] = intercept

    if cfg.image_processing.mask_separate is True:
        image_list = ["BMD", "SEG", "CORTMASK", "TRABMASK"]
        threads = []
        lock = threading.Lock()
        for item in image_list:
            t = threading.Thread(
                target=imutils.read_aim, args=(item, filenames, bone, lock)
            )
            threads.append(t)
            sleep(0.1)  # to avoid overloading the print statements
            t.start()
        for t in threads:
            t.join()
    else:
        image_list = ["BMD", "SEG"]
        for _, item in enumerate(image_list):
            bone = imutils.read_aim(item, filenames, bone, lock=None)
        bone = imutils.read_aim_mask_combined("MASK", filenames, bone)

    # image_list = ["BMD", "SEG", "CORTMASK", "TRABMASK"]
    if cfg.registration.registration is True:
        for _, item in enumerate(image_list):
            bone = imutils.adjust_image_size(item, bone, cfg, imutils.CropType.crop)
    else:
        pass

    # Save images with colorbar
    imutils.save_images_with_colorbar(cfg, sample, bone)

    # 3 Prepare material mapping
    # ---------------------------------------------------------------------------------

    IMTYPE = cfg.image_processing.imtype
    io_utils.print_mem_usage()
    BVTVscaled, BMDscaled, BVTVraw = preprocessing.calculate_bvtv(
        bone["Scaling"],
        bone["Slope"],
        bone["Intercept"],
        bone["BMD_array"],
        bone["CORTMASK_array"],
        bone["TRABMASK_array"],
        cfg,
        IMTYPE,
    )

    bone["BVTVscaled"] = BVTVscaled
    bone["BMDscaled"] = BMDscaled
    bone["BVTVraw"] = BVTVraw

    del BVTVscaled, BMDscaled, BVTVraw
    gc.collect()

    if cfg.mesher.meshing == "spline":
        cort_mask_np = bone["CORTMASK_array"]
        trab_mask_np = bone["TRABMASK_array"]
        masks = [cort_mask_np, trab_mask_np]
        mask_names = ["CORTMASK", "TRABMASK"]
        for mask, mask_name in zip(masks, mask_names):
            sitk_image = sitk.GetImageFromArray(mask)
            sitk_image = sitk.PermuteAxes(sitk_image, [2, 1, 0])
            sitk_image = sitk.Flip(sitk_image, [False, True, False])
            sitk_image.SetSpacing(bone["Spacing"])
            cortmask_path = (
                Path(cfg.paths.aimdir)
                / cfg.simulations.folder_id[sample]
                / f"{sample}_{mask_name}.mhd"
            )
            sitk.WriteImage(sitk_image, cortmask_path)
            if mask_name == "CORTMASK":
                sitk_image_cort = sitk_image
            else:
                pass

        # append sample filename to config_mesh["img_settings"]
        sample_n = str(sample)
        io_utils.hydra_update_cfg_key(cfg, "img_settings.img_basefilename", sample_n)
        cfg.img_settings.img_basefilename = sample_n
        mesh = HexMesh(
            cfg.meshing_settings,
            cfg.img_settings,
            sitk_image=sitk_image_cort,
        )

        (
            nodes,
            elms,
            nb_nodes,
            centroids_cort,
            centroids_trab,
            elm_vol_cort,
            elm_vol_trab,
            radius_roi_cort,
            radius_roi_trab,
            bnds_bot,
            bnds_top,
            reference_point_coord,
        ) = mesh.mesher()

        bone["nodes"] = nodes
        bone["elms"] = elms
        bone["nb_nodes"] = nb_nodes
        bone["degrees_of_freedom"] = len(nodes) * 6
        bone["elms_centroids_cort"] = centroids_cort
        bone["elms_centroids_trab"] = centroids_trab
        bone["elms_vol_cort"] = elm_vol_cort
        bone["elms_vol_trab"] = elm_vol_trab
        bone["radius_roi_cort"] = radius_roi_cort
        bone["radius_roi_trab"] = radius_roi_trab
        bone["bnds_bot"] = bnds_bot
        bone["bnds_top"] = bnds_top
        bone["reference_point_coord"] = reference_point_coord

        bone["elsets"] = []
        if "FEelSize" not in bone or bone["FEelSize"]:
            bone["FEelSize"] = (
                int(round(cfg.mesher.element_size / spacing[0])) * bone["Spacing"]
            )
        # old voxel-based mesh settings
        CoarseFactor = bone["FEelSize"][0] / bone["Spacing"][0]
        bone["CoarseFactor"] = CoarseFactor
        BVTVscaled_shape = bone["BVTVscaled"].shape
        bone["MESH"] = np.ones(
            ([int(dim) for dim in np.floor(np.array(BVTVscaled_shape) / CoarseFactor)])
        )

    # 4 Material mapping
    # ---------------------------------------------------------------------------------
    # Compute MSL kernel list
    if cfg.homogenization.fabric_type == "local":
        logger.info("Computing local MSL kernel list")
        bone = preprocessing.compute_msl_spline(bone, cfg)
    elif cfg.homogenization.fabric_type == "global":
        logger.info("Computing global MSL kernel list")
        pass
    else:
        raise ValueError("Fabric type not recognised")

    if cfg.mesher.meshing == "spline":
        (
            bone,
            abq_dictionary,
            abq_inp_path,
        ) = material_mapping.material_mapping_spline(
            bone,
            cfg,
            filenames,
        )
        bone["abq_inp_path"] = abq_inp_path
    else:
        raise ValueError("Meshing type not recognised (ghost layer mode 1)")

    # 5 Compute and store summary and performance variables
    # ---------------------------------------------------------------------------------
    logger.info("Computing summary variables")
    summary_variables = preprocessing.set_summary_variables(bone)
    # io_utils.log_summary(bone, cfg, filenames, summary_variables)
    bone = dict(list(bone.items()) + list(summary_variables.items()))
    logger.info("Summary variables computed")

    if cfg.mesher.meshing == "spline":
        imutils.plot_MSL_fabric_spline(cfg, abq_dictionary, sample)
    else:
        raise NotImplementedError("Meshing type not recognised")
    return bone, abq_inp_path
