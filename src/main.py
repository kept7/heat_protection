from os import getenv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pandas import read_excel, DataFrame, ExcelWriter
import libs.lpre_heat_protection as hp

"""
Init data:
    - engine type
    - thrust, chambers count
    - fuel+ox, mass flow rate
    - Km, alpha
    - comb cham pres
    - nozzle exit pres
    - comb cham temp
    - diametr of nozzle, etcs
"""


def main_programm() -> None:
    # getting x, D coordinates and D_kp and F_kp from excel file
    X_D_PATH_FILE = get_env_path("X_D_PATH_FILE")
    RESULT_PATH_FILE = get_env_path("RESULT_PATH_FILE")

    x_coord_list = []
    d_list = []
    x_coord_list, d_list = get_xlsx_data(X_D_PATH_FILE, x_coord_list, d_list)

    D_kp = x_coord_list.pop(0)
    F_kp = d_list.pop(0)

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
    cooling_path_params = hp.cooling_path_params(d_list)
    column_names = [
        "n_p",
        "t",
        "t_N",
        "f",
        "d_г",
        "b",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "1.3.2", "a", cooling_path_params)

    # getting heat flows parameters (header 2.2 - manual)
    heat_flows_res = hp.heat_flows_calc(x_coord_list, d_list)
    column_names = [
        "x",
        "lymbda",
        "beta",
        "S",
        "T_отн_ст",
        "q * 10^(-6)",
        "q_л * 10^(-6)",
        "q_г * 10^(-6)"
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "2.2", "a", heat_flows_res)


def get_env_path(env_name: str) -> str:
    dotenv_path = Path("../.env")
    load_dotenv(dotenv_path=dotenv_path)
    path_file = getenv(env_name)

    return path_file


def get_xlsx_data(
    path_file: str, first_col: List[None], second_col: List[None]
) -> List[float]:
    df = read_excel(path_file, header=None)

    for _, row in df.iterrows():
        first_col.append(round(row.tolist()[0], 6))
        second_col.append(round(row.tolist()[1], 6))

    return first_col, second_col


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
