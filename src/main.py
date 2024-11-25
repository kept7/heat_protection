from os import getenv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pandas import read_excel, DataFrame, ExcelWriter
from numpy import pi, radians
import libs.lpre_heat_protection as hp


def main_programm() -> None:
    # getting initial data from excel file
    X_D_PATH_FILE = get_env_path("X_D_PATH_FILE")
    RESULT_PATH_FILE = get_env_path("RESULT_PATH_FILE")

    x_coord_list = []
    d_list = []
    W_list = []
    Cp_t_og_list = []
    Cp_t_ct_list = []
    p_list = []
    mu_list = []
    (
        x_coord_list,
        d_list,
        W_list,
        p_list,
        mu_list,
        D_kp,
        mode,
        h,
        delta_ct,
        delta_p,
        delta_ct_HAP,
        beta,
        gamma,
        t_N_min,
        k,
        Pr,
        alpha,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        m_t,
        T_k,
    ) = get_xlsx_data(
        X_D_PATH_FILE,
        x_coord_list,
        d_list,
        W_list,
        Cp_t_og_list,
        Cp_t_ct_list,
        p_list,
        mu_list,
    )

    index_kp = d_list.index(D_kp)
    F_kp = (pow((d_list[index_kp]), 2)) * pi / 4

    # getting parameters of the flow part of the chamber (header 1.3.1 - manual)
    ch_resol_res = hp.chamber_params(x_coord_list, d_list, D_kp, F_kp)
    column_names = [
        "x",
        "D",
        "D OTH",
        "F",
        "F OTH",
        "delta(x)",
        "delta(xs)",
        "delta(S)",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "1.3.1", "w", ch_resol_res)

    # getting cooling path parameters (header 1.3.2 - manual)
    cooling_path_params = hp.cooling_path_params(
        d_list, mode, h, delta_ct, delta_p, delta_ct_HAP, beta, gamma, t_N_min
    )
    column_names = [
        "n_p",
        "t",
        "t_N",
        "f",
        "d_г",
        "b",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "1.3.2", "a", cooling_path_params)

    # # getting heat flows parameters (header 2.2 - manual)
    F_kc = pow(d_list[0], 2) * pi / 4
    heat_flows_res = hp.heat_flows_calc(
        x_coord_list,
        d_list,
        W_list,
        Cp_t_og_list,
        Cp_t_ct_list,
        mu_list,
        p_list,
        F_kc,
        F_kp,
        index_kp,
        k,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        Pr,
        T_k,
    )
    # column_names = [
    #     "x",
    #     "lymbda",
    #     "beta",
    #     "S",
    #     "T_отн_ст",
    #     "q_к * 10^(-6)",
    #     "q_л * 10^(-6)",
    #     "q_сум * 10^(-6)"
    # ]
    # write_xlsx_data(RESULT_PATH_FILE, column_names, "2.2", "a", heat_flows_res)


def get_env_path(env_name: str) -> str:
    dotenv_path = Path("../.env")
    load_dotenv(dotenv_path=dotenv_path)
    path_file = getenv(env_name)

    return path_file


def get_xlsx_data(
    path_file: str,
    first_list: List[None],
    second_list: List[None],
    third_list: List[None],
    fourth_list: List[None],
    fifth_list: List[None],
    sixth_lsit: List[None],
    seventh_list: List[None],
) -> List[float]:
    df = read_excel(path_file, header=None)

    for index, row in df.iterrows():
        if index > 0:
            first_list.append(round(row.tolist()[0], 6))
            second_list.append(round(row.tolist()[1], 6))
            third_list.append(round(row.tolist()[11], 6))
            fourth_list.append(round(row.tolist()[19], 6))
            fifth_list.append(round(row.tolist()[20], 6))
            sixth_lsit.append(round(row.tolist()[21], 6))
            seventh_list.append(round(row.tolist()[22], 6))
        if index == 1:
            D_kp = round(row.tolist()[2], 6)
            mode = row.tolist()[3]
            h = row.tolist()[4]
            delta_ct = row.tolist()[5]
            delta_p = row.tolist()[6]
            delta_ct_HAP = row.tolist()[7]
            beta = radians(row.tolist()[8])
            gamma = radians(row.tolist()[9])
            t_N_min = row.tolist()[10]
            k = row.tolist()[12]
            Pr = row.tolist()[13]
            alpha = row.tolist()[14]
            T_ct_g = row.tolist()[15]
            T_ct_o = row.tolist()[16]
            mu_og = row.tolist()[17]
            R_og = row.tolist()[18]
            m_t = row.tolist()[23]
            T_k = row.tolist()[24]

    return (
        first_list,
        second_list,
        third_list,
        fourth_list,
        fifth_list,
        sixth_lsit,
        seventh_list,
        D_kp,
        mode,
        h,
        delta_ct,
        delta_p,
        delta_ct_HAP,
        beta,
        gamma,
        t_N_min,
        k,
        Pr,
        alpha,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        m_t,
        T_k,
    )


def write_xlsx_data(
    PATH_FILE: str,
    column_names: str,
    sheet_number: str,
    mode: str,
    data: List[List[float]],
) -> None:
    df = DataFrame(data, columns=column_names)
    with ExcelWriter(PATH_FILE, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_number, index=False)


if __name__ == "__main__":
    """
    Before running the code cd to src folder
    """

    main_programm()

    hp.hello_world()
