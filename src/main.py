from os import getenv
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pandas import read_excel, DataFrame, ExcelWriter
from matplotlib.pyplot import plot, show
from scipy.interpolate import interp1d
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
    # getting x and D coordinates from excel file
    X_D_PATH_FILE = get_env_path("X_D_PATH_FILE")

    x_coord_list = []
    d_list = []

    x_coord_list, d_list = get_xlsx_data(X_D_PATH_FILE, x_coord_list, d_list)

    # getting parameters of the flow part of the chamber (header 1.3.1 - manual)
    D_kp = 0.220165
    F_kp = 0.03807
    ch_res_res = hp.chamber_params(x_coord_list, d_list, D_kp, F_kp)

    # writing the results to excel file
    RESULT_PATH_FILE = get_env_path("RESULT_PATH_FILE")
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
    write_xlsx_data(RESULT_PATH_FILE, column_names, "first_sheet", ch_res_res)

    # r_from_d = [i / 2 for i in d_list]
    # y_interp = interp1d(x_coord_list, r_from_d, kind="linear")
    # # xnew = arange(x_coord_list[1], x_coord_list[-1], x_coord_list[-1] / 32)
    # test = [(x_coord_list[i+1] - x_coord_list[i]) for i, el in enumerate(x_coord_list[1:])]
    # print(test)
    # ynew = y_interp(test)
    # print(ynew)

    # plot(x_coord_list[1:], ynew, 'o--')
    # show()


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
    PATH_FILE: str, column_names: str, sheet_number: str, result: List[List[int]]
) -> None:
    df = DataFrame(result, columns=column_names)
    writer = ExcelWriter(PATH_FILE, engine="xlsxwriter")
    df.to_excel(writer, sheet_name=sheet_number, index=False)
    writer.close()


if __name__ == "__main__":
    """
    Before running the code cd to src folder
    """

    main_programm()

    hp.hello_world()
