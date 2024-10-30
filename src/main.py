from os import getenv, path
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from pandas import read_excel
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
    X_D_PATH_FILE = get_env_path("X_D_PATH_FILE")

    # getting x and D coordinates
    x_coord = []
    D_coord = []

    x_coord, D_coord = xlsx_data(X_D_PATH_FILE, x_coord, D_coord)

    print(x_coord)
    print(D_coord)

def get_env_path(env_name) -> str:
    dotenv_path = Path("../.env")
    load_dotenv(dotenv_path=dotenv_path)
    path_file = getenv(env_name)

    return path_file

def xlsx_data(path_file, first_col, second_col) -> List[float]:
    df = read_excel(path_file, header=None)

    for _, row in df.iterrows():
        first_col.append(row.tolist()[0])
        second_col.append(row.tolist()[1])

    return first_col, second_col

if __name__ == "__main__":
    """
    Before running the code cd to src folder
    """
    main_programm()

    hp.hello_world()