import numpy as np
import pandas as pd


# flake8: noqa: E501
def remove_empty_entries_list(list):
    while "" in list:
        list.remove("")
    list[-1] = list[-1][:-1]
    return list


def datfilereader_6d(infilename):
    outfilename = infilename.replace(".dat", ".txt")

    # Read the .dat file

    # Read the .dat file
    with open(infilename, "r") as infile:
        lines = infile.readlines()

    # Lists used
    ref_nodedata = []
    disp = []
    force = []
    # Ref node outputs
    # Displacement vector U
    U1 = " "
    U2 = " "
    U3 = " "
    UR1 = " "
    UR2 = " "
    UR3 = " "
    # Force vector R
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

            # Extract different U components
            member1_red = remove_empty_entries_list(member1)
            U1 = float(member1_red[1])
            U2 = float(member1_red[2])
            U3 = float(member1_red[3])
            UR1 = float(member1_red[4])
            UR2 = float(member1_red[5])
            UR3 = float(member1_red[6])

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

            disp.append([U1, U2, U3, UR1, UR2, UR3])
            force.append([RF1, RF2, RF3, RM1, RM2, RM3])

            ref_nodedata.append(
                str(j)
                + ", "
                + str(U1)
                + ", "
                + str(U2)
                + ", "
                + str(U3)
                + ", "
                + str(UR1)
                + ", "
                + str(UR2)
                + ", "
                + str(UR3)
                + ", "
                + str(RF1)
                + ", "
                + str(RF2)
                + ", "
                + str(RF3)
                + ", "
                + str(RM1)
                + ", "
                + str(RM2)
                + ", "
                + str(RM3)
            )
    # Create the output files
    with open(outfilename, "w") as out:
        out.write(
            """
*********************************************************
* Displacement and reaction force of the reference node *
*                 Ghislain MAQUER 2011                  *
*********************************************************
inc U1 U2 U3 UR1 UR2 UR3 RF1 RF2 RF3 RM1 RM2 RM3
0,   0,  0,  0,  0,   0,   0,   0,   0,   0,   0,   0,   0
"""
        )
        for i in range(0, len(ref_nodedata)):
            out.write(ref_nodedata[i] + "\n")
    return outfilename


def datfilereader_force(infilename):
    dat_filename = infilename

    outfilename = infilename.with_suffix(".txt")

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
            except ValueError:
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
            except ValueError:
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
    # Create the output files
    with open(outfilename, "w") as out:
        out.write(
            """
*********************************************************
* Displacement and reaction force of the reference node *
*                 Ghislain MAQUER 2011                  *
*********************************************************
inc U1 U2 U3 UR1 UR2 UR3 RF1 RF2 RF3 RM1 RM2 RM3 CF1 CF2 CF3 CM1 CM2 CM3
0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
"""
        )
        for i in range(0, len(ref_nodedata)):
            out.write(ref_nodedata[i] + "\n")
    return ref_nodedata


def __stiffness__(ref_nodedata):
    FZ = ref_nodedata["RF3"].astype(float)
    DZ = ref_nodedata["U3"].astype(float)

    try:
        k = FZ[0] / DZ[0]
    except Exception:
        df = FZ[2] - FZ[1]
        dd = DZ[2] - DZ[1]
        k = df / dd
    print("    - Stiffness = ", str(k), " N/mm")
    return k, FZ, DZ


def __yield_point__(height, k, FZ, DZ):
    """
    height: sample height
    k: stiffness
    FZ: Reaction Force in Z direction
    DZ: Displacement in Z direction
    """
    h02 = height * 0.002
    intersect02 = -h02 * k
    F_offsetrule = np.array(DZ) * k + intersect02
    for i in range(1, len(FZ)):
        if FZ[i] < F_offsetrule[i]:
            s_temp = (FZ[i] - FZ[i - 1]) / (DZ[i] - DZ[i - 1])
            int_temp = FZ[i - 1] - s_temp * DZ[i - 1]
            disp_yield = (int_temp - intersect02) / (k - s_temp)
            Fyield = s_temp * disp_yield + int_temp
            break
    try:
        print("    - F yield = ", str(Fyield), " N")
    except UnboundLocalError:
        Fyield = 1e6
        print("    - F yield = ", str(Fyield), " N")
    try:
        print("    - Displacement at F yield = ", str(disp_yield), " mm")
    except UnboundLocalError:
        disp_yield = 1e6
        print("    - Displacement at F yield = ", str(disp_yield), " mm")
    return Fyield, disp_yield


def __calc_mech_props__(ref_nodedata, thickness_s=30.6):
    stiffness, _, _ = __stiffness__(ref_nodedata)
    U3 = np.abs(ref_nodedata["U3"].values.astype(float))
    RF3 = np.abs(ref_nodedata["RF3"].values.astype(float))
    yield_force, yield_displacement = __yield_point__(thickness_s, stiffness, RF3, U3)
    max_force = np.max(RF3)
    # disp_at_max_force: displacement at same index as max force
    index_max_force = np.where(RF3 == max_force)[0][0]
    disp_at_max_force = U3[index_max_force]

    return stiffness, yield_force, yield_displacement, max_force, disp_at_max_force


def parse_and_calculate_stiffness_yield_force(
    dat_filename: str, thickness: float = 30.6
):
    print("Reading datfile: ", dat_filename)
    ref_nodedata = datfilereader_force(dat_filename)
    column_names = [
        "inc",
        "U1",
        "U2",
        "U3",
        "UR1",
        "UR2",
        "UR3",
        "RF1",
        "RF2",
        "RF3",
        "RM1",
        "RM2",
        "RM3",
        "CF1",
        "CF2",
        "CF3",
        "CM1",
        "CM2",
        "CM3",
    ]
    data_processed_s = [line.split() for line in ref_nodedata]
    ref_nodedata_processed = pd.DataFrame(data_processed_s, columns=column_names)

    stiffness, yield_force, yield_displacement, max_force, disp_at_max_force = (
        __calc_mech_props__(ref_nodedata_processed, thickness)
    )
    return stiffness, yield_force, yield_displacement, max_force, disp_at_max_force


def main():
    path2dat = "/home/sp20q110/HFE/04_SIMULATIONS/443_L_73_F/C0003101_03.dat"
    stiffness, yield_force, yield_displacement, max_force = (
        parse_and_calculate_stiffness_yield_force(path2dat, thickness=30.6)
    )
    print("Stiffness: ", stiffness)
    print("Yield Force: ", yield_force)
    print("Yield Displacement: ", yield_displacement)
    print("Max Force: ", max_force)


if __name__ == "__main__":
    main()
