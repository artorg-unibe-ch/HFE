import logging
from csv import writer
from pathlib import Path

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def remove_empty_entries_list(listname: list) -> list:
    while "" in listname:
        listname.remove("")
    listname[-1] = listname[-1][:-1]
    return listname


def datfilereader_psl(cfg, sample: str, optim: dict, loadcase: str) -> dict:
    """
    Function same as in  PSL_Fast, DO NOT CHANGE!
    Reads forces and displacements from Abaqus .dat file and stores them in optim dictionary, as well as a
    structured text file
    @param config: configuration parameters dictionary
    @param sample: sample number
    @param optim: optimization dictionary for results
    @param loadcase: load case string
    @return: optim dict
    """
    dat_filename = (
        cfg.paths.feadir
        + "/"
        + cfg.simulations.folder_id[sample]
        + "/"
        + sample
        + "_"
        + cfg.version.current_version[0:2]
        + ".dat"
    )

    outfilename = dat_filename.replace(".dat", ".txt")

    # Read the .dat file
    with open(dat_filename, "r") as infile:
        lines = infile.readlines()

    # Lists used
    ref_nodedata = []
    disp = []
    force = []
    cforce = []
    # Ref node outputs
    # Displacement vector U
    U1 = " "
    U2 = " "
    U3 = " "
    UR1 = " "
    UR2 = " "
    UR3 = " "
    # Rectionforce vector R
    RF1 = " "
    RF2 = " "
    RF3 = " "
    RM1 = " "
    RM2 = " "
    RM3 = " "
    # Rectionforce vector R
    RF1 = " "
    RF2 = " "
    RF3 = " "
    RM1 = " "
    RM2 = " "
    RM3 = " "
    j = 0
    for i in range(0, len(lines)):
        if lines[i].find("U3") > -1:
            j = j + 1  # value of increment
            # Split U and R lines
            member1 = lines[i + 3].split(" ")  # value of U3
            member2 = lines[i + 12].split(" ")  # value of RF3
            member3 = lines[i + 21].split(" ")  # value of CF3

            # Extract different U components
            member1_red = remove_empty_entries_list(member1)
            U1 = float(member1_red[1])
            U2 = float(member1_red[2])
            U3 = float(member1_red[3])
            UR1 = float(member1_red[4])
            UR2 = float(member1_red[5])
            UR3 = float(member1_red[6])

            try:
                # Extract different R components
                member2_red = remove_empty_entries_list(member2)
                # Clean member2_pass from "+" and replace with "E+"
                for k in range(0, len(member2_red)):
                    if member2_red[k].find("+") != -1:
                        if member2_red[k].find("E+") != -1:
                            pass
                        else:
                            member2_red[k] = member2_red[k].replace("+", "E+")
                RF1 = float(member2_red[1])
                RF2 = float(member2_red[2])
                RF3 = float(member2_red[3])
                RM1 = float(member2_red[4])
                RM2 = float(member2_red[5])
                RM3 = float(member2_red[6])
            except Exception:
                RF1 = 0.0
                RF2 = 0.0
                RF3 = 0.0
                RM1 = 0.0
                RM2 = 0.0
                RM3 = 0.0

            try:
                # Extract different C components
                member3_red = remove_empty_entries_list(member3)
                # Clean member2_pass from "+" and replace with "E+"
                for k in range(0, len(member3_red)):
                    if member3_red[k].find("+") != -1:
                        if member3_red[k].find("E+") != -1:
                            pass
                        else:
                            member3_red[k] = member3_red[k].replace("+", "E+")
                CF1 = float(member3_red[1])
                CF2 = float(member3_red[2])
                CF3 = float(member3_red[3])
                CM1 = float(member3_red[4])
                CM2 = float(member3_red[5])
                CM3 = float(member3_red[6])
            except Exception:
                CF1 = 0.0
                CF2 = 0.0
                CF3 = 0.0
                CM1 = 0.0
                CM2 = 0.0
                CM3 = 0.0

            disp.append([U1, U2, U3, UR1, UR2, UR3])
            force.append([RF1, RF2, RF3, RM1, RM2, RM3])
            cforce.append([CF1, CF2, CF3, CM1, CM2, CM3])

            ref_nodedata.append(
                str(j)
                + " "
                + str(U1)
                + " "
                + str(U2)
                + " "
                + str(U3)
                + " "
                + str(UR1)
                + " "
                + str(UR2)
                + " "
                + str(UR3)
                + " "
                + str(RF1)
                + " "
                + str(RF2)
                + " "
                + str(RF3)
                + " "
                + str(RM1)
                + " "
                + str(RM2)
                + " "
                + str(RM3)
                + " "
                + str(CF1)
                + " "
                + str(CF2)
                + " "
                + str(CF3)
                + " "
                + str(CM1)
                + " "
                + str(CM2)
                + " "
                + str(CM3)
            )
    # Create the output files
    with open(outfilename, "w") as out:
        out.write(
            """
*********************************************************
* Displacement and reaction force of the reference node *
*                 Denis SCHENK 2011                  *
*********************************************************
inc U1 U2 U3 UR1 UR2 UR3 RF1 RF2 RF3 RM1 RM2 RM3 CF1 CF2 CF3 CM1 CM2 CM3
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
"""
        )
        for i in range(0, len(ref_nodedata)):
            out.write(ref_nodedata[i] + "\n")

    optim["disp_" + loadcase] = disp
    optim["force_" + loadcase] = force
    optim["conc_force_" + loadcase] = cforce

    return optim


def write_data_summary(
    cfg,
    optim: dict,
    bone: dict,
    sample: str,
    mesh_parameters_dict: dict,
    DOFs: int,
    time_sim: float,
) -> None:
    """
    Function adapted for accurate no psl (only FZ_MAX load case)
    Function that writes a summary csv file with all OPTIM parameters, for easy postprocessing in R studio
    File is stored in summary folder.
    @param config:
    @param optim:
    @return:
    """

    # Make summary file folder if not yet existing
    summary_path = Path(cfg.paths.sumdir)
    summary_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory '{summary_path}' created")

    # Create summary file if not yet existing
    filename = summary_path / (cfg.version.current_version + "_data_summary.csv")
    try:
        field_names_dict = [
            sample,
            optim["max_force_FZ_MAX"],
            optim["disp_at_max_force_FZ_MAX"],
            optim["stiffness_FZ_MAX"],
            bone["TOT_mean_BMC_image"],
            bone["TOT_simulation_BMC_FE_tissue_orig_ROI"],
            bone["TOT_simulation_BMC_FE_tissue_ROI"],
            bone["TOT_BMC_ratio_ROI"],
            bone["mean_BMD_SEG_CORTorig"],
            bone["mean_BMD_SEG_TRABorig"],
            bone["mean_BMD_SEG_CORTscaled"],
            bone["mean_BMD_SEG_TRABscaled"],
            bone["BV_CORT_SEG"],
            bone["BV_TRAB_SEG"],
            bone["nel_CORT"],
            bone["nel_TRAB"],
            bone["nel_MIXED"],
            cfg["bvtv_scaling"],
            cfg["bvtv_slope"],
            cfg["bvtv_intercept"],
            bone["mean_BVTV_seg"],
            bone["mean_BVTVd_scaled"],
            bone["mean_BVTVd_raw"],
        ]

        field_names_titles = [
            "Sample",
            "max_force_FZ_MAX",
            "disp_at_max_force_FZ_MAX",
            "stiffness_1D_FZ_MAX",
            "BMC_tissue",
            "BMC_tissue_ROI_orig",
            "BMC_tissue_ROI_corrected",
            "BMC_ratio_corrected",
            "mean_tissue_BMD_CORT_orig",
            "mean_tissue_BMD_TRAB_orig",
            "mean_tissue_BMD_CORT_scaled",
            "mean_tissue_BMD_TRAB_scaled",
            "BV_CORT_SEG",
            "BV_TRAB_SEG",
            "nel_CORT",
            "nel_TRAB",
            "nel_MIXED",
            "Hosseini_scaling",
            "manual_scaling_slope",
            "manual_scaling_intercept",
            "mean_BVTV_seg",
            "mean_BVTVd_scaled",
            "mean_BVTVd_raw",
        ]
    except Exception:
        # field_names_dict = [
        #     sample,
        #     optim["max_force_FZ_MAX"],
        #     optim["disp_at_max_force_FZ_MAX"],
        #     optim["stiffness_FZ_MAX"],
        # ]

        # field_names_titles = [
        #     "Sample",
        #     "max_force_FZ_MAX",
        #     "disp_at_max_force_FZ_MAX",
        #     "stiffness_1D_FZ_MAX",
        # ]

        field_names_dict = [
            sample,
            DOFs,
            time_sim,
            mesh_parameters_dict["n_elms_longitudinal"],
            mesh_parameters_dict["n_elms_transverse_trab"],
            mesh_parameters_dict["n_elms_transverse_cort"],
            mesh_parameters_dict["n_elms_radial"],
            optim["max_force_FZ_MAX"],
            optim["disp_at_max_force_FZ_MAX"],
            optim["stiffness_FZ_MAX"],
            optim["yield_force_FZ_MAX"],
            optim["yield_disp_FZ_MAX"],
            bone["trab_avg_DA"],
        ]

        field_names_titles = [
            "Sample",
            "DOFs",
            "simulation_time",
            "n_elms_longitudinal",
            "n_elms_transverse_trab",
            "n_elms_transverse_cort",
            "n_elms_radial",
            "max_force_FZ_MAX",
            "disp_at_max_force_FZ_MAX",
            "stiffness_1D_FZ_MAX",
            "yield_force_FZ_MAX",
            "yield_disp_FZ_MAX",
            "trabecular_average_DA",
        ]

    def append_list_as_row(filename: str, list_of_elem: list) -> None:
        with open(filename, "a+", newline="") as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)

    if filename.exists():
        append_list_as_row(filename, field_names_dict)
    else:
        file = open(filename, "a+", newline="")
        csv_writer = writer(file)
        csv_writer.writerow(field_names_titles)
        file.close()
        append_list_as_row(filename, field_names_dict)

    return None
