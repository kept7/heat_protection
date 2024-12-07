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
    (
        x_coord_list,
        d_list,
        D_kp,
        mode,
        h,
        delta_ct,
        delta_p,
        delta_ct_HAP,
        beta,
        gamma,
        t_N_min,
        lymbda_material,
        k,
        Pr,
        alpha,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        Cp_t_og,
        Cp_t_ct,
        phi,
        p_k,
        T_k,
        epsilon_h2o,
        epsilon_co2,
        epsilon_ct,
        m_oxl,
        coolant_type,
        T_init,
        i_enter,
        i_exit,
        T_ycl,
        p_init,
        delta_wall,
    ) = get_xlsx_data(X_D_PATH_FILE, x_coord_list, d_list)

    index_kp = d_list.index(D_kp)
    F_kp = (pow((d_list[index_kp]), 2)) * pi / 4

    # getting parameters of the flow part of the chamber (header 1.3.1 - manual)
    ch_resol_res, Delta_S, Delta_xs_list = hp.chamber_params(
        x_coord_list, d_list, D_kp, F_kp
    )
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
    cooling_path_params, t_list, f_list, d_g_list, h_p = hp.cooling_path_params(
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

    # getting heat flows parameters (header 2.2 - manual)
    q_sum_list = []
    heat_flows_res, S_list, q_l_list, q_k_list, q_sum_list = hp.heat_flows_calc(
        x_coord_list,
        d_list,
        index_kp,
        k,
        Cp_t_og,
        Cp_t_ct,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        Pr,
        p_k,
        T_k,
        phi,
        epsilon_h2o,
        epsilon_co2,
        epsilon_ct,
    )

    column_names = [
        "x",
        "lymbda",
        "beta",
        "S",
        "T_отн_ст",
        "q_к * 10^(-6)",
        "q_л * 10^(-6)",
        "q_сум * 10^(-6)",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "2.2", "a", heat_flows_res)

    # Calculation of heat transfer in the cooling path (header 3.1, 3.2, 3.3 - manual)
    (
        res,
        T_oxl_list,
        cp_oxl_list,
        alpha_oxl_list,
        eta_p_list,
        rho_oxl_list,
        U_oxl_list,
        mu_oxl_list,
    ) = hp.heat_in_cooling_path_calc(
        q_sum_list,
        Delta_S,
        f_list,
        d_g_list,
        t_list,
        i_enter,
        i_exit,
        T_init,
        m_oxl,
        coolant_type,
        delta_p,
        h_p,
        lymbda_material,
        beta,
        p_init * 1000000,
    )

    column_names = [
        "Т_охл",
        "delta_Cp_OTH",
        "Cp_охл",
        "lymbda_охл",
        "mu_охл",
        "К_охл",
        "U_охл",
        "alpha_охл",
        "E",
        "eta_р",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "3.3", "a", res)

    # Calculation temperature_first_approx (header 4.2 - manual)
    res, first_appr_T_ct_g_list = hp.temperature_approx(
        x_coord_list,
        q_l_list,
        q_k_list,
        T_oxl_list,
        alpha_oxl_list,
        eta_p_list,
        T_ct_o,
        T_ycl,
        delta_ct,
        lymbda_material,
    )

    column_names = [
        "x",
        "Т ст г",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "4.2", "a", res)

    # # Calculation temperature with local curtain if needed (header 4.3 - manual)
    # local_curtain = 0
    # for temp in appr_T_ct_g_list:
    #     if temp >= T_ycl:
    #         local_curtain = 1

    # if local_curtain:
    #     res = hp.temperature_with_local_curtain(

    # )

    # column_names = [
    #
    # ]
    # write_xlsx_data(RESULT_PATH_FILE, column_names, "4.3", "a", res)

    # Calculation temperature_second_approx and temperature of coolant wall (header 4.4, 4.5 - manual)
    q_k_sec_approx_list, S_list_sec_approx = hp.temperature_second_approx(
        S_list,
        q_k_list,
        cp_oxl_list,
        first_appr_T_ct_g_list,
        mu_og,
        R_og,
        T_ct_o,
    )

    q_sum_sec_approx_list = [
        q_k_sec_approx_list[i] + q_l_val for i, q_l_val in enumerate(q_l_list)
    ]

    (
        res1,
        T_oxl_sec_approx_list,
        cp_oxl_sec_approx_list,
        alpha_oxl_sec_approx_list,
        eta_p_sec_approx_list,
        rho_oxl_sec_approx_list,
        U_oxl_sec_approx_list,
        mu_oxl_sec_approx_list,
    ) = hp.heat_in_cooling_path_calc(
        q_sum_sec_approx_list,
        Delta_S,
        f_list,
        d_g_list,
        t_list,
        i_enter,
        i_exit,
        T_init,
        m_oxl,
        coolant_type,
        delta_p,
        h_p,
        lymbda_material,
        beta,
        p_init * 1000000,
    )

    res2, second_appr_T_ct_g_list = hp.temperature_approx(
        x_coord_list,
        q_l_list,
        q_k_sec_approx_list,
        T_oxl_sec_approx_list,
        alpha_oxl_sec_approx_list,
        eta_p_sec_approx_list,
        T_ct_o,
        T_ycl,
        delta_ct,
        lymbda_material,
    )

    temperature_coolant_wall = hp.calculation_temperature_coolant_wall(
        second_appr_T_ct_g_list,
        q_sum_sec_approx_list,
        delta_ct,
        lymbda_material,
    )

    result = [
        [
            x_coord_list[i],
            S_list_sec_approx[i],
            q_k_sec_approx_list[i],
            q_sum_sec_approx_list[i],
            T_oxl,
            second_appr_T_ct_g_list[i],
            temperature_coolant_wall[i],
        ]
        for i, T_oxl in enumerate(T_oxl_sec_approx_list)
    ]

    column_names = [
        "x",
        "S''",
        "q''",
        "q_сум''",
        "T_охл''",
        "T_ст.г''",
        "T_ст_охл",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "4.5", "a", result)

    # Calculation of coolant pressure losses in the cooling tract (header 5 - manual)
    res = hp.calc_coolant_pressure_losses(
        Delta_xs_list,
        d_g_list,
        rho_oxl_sec_approx_list,
        U_oxl_sec_approx_list,
        mu_oxl_sec_approx_list,
        beta,
        delta_wall,
        t_N_min,
        delta_p,
        h_p,
    )

    column_names = [
        "Re",
        "delta_отн",
        "Re_гр",
        "Zeta",
        "l",
        "delta_p",
    ]
    write_xlsx_data(RESULT_PATH_FILE, column_names, "5", "a", res)


def get_env_path(env_name: str) -> str:
    dotenv_path = Path("../.env")
    load_dotenv(dotenv_path=dotenv_path)
    path_file = getenv(env_name)

    return path_file


def get_xlsx_data(
    path_file: str,
    first_list: List[None],
    second_list: List[None],
) -> List[float]:
    df = read_excel(path_file, header=None)

    for index, row in df.iterrows():
        if index > 0:
            first_list.append(round(row.tolist()[0], 6))
            second_list.append(round(row.tolist()[1], 6))
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
            lymbda_material = row.tolist()[11]
            k = row.tolist()[12]
            Pr = row.tolist()[13]
            alpha = row.tolist()[14]
            T_ct_g = row.tolist()[15]
            T_ct_o = row.tolist()[16]
            mu_og = row.tolist()[17]
            R_og = row.tolist()[18]
            Cp_t_og = row.tolist()[19]
            Cp_t_ct = row.tolist()[20]
            phi = row.tolist()[21]
            p_k = row.tolist()[22]
            T_k = row.tolist()[23]
            epsilon_h2o = row.tolist()[24]
            epsilon_co2 = row.tolist()[25]
            epsilon_ct = row.tolist()[26]
            m_oxl = row.tolist()[27]
            coolant_type = row.tolist()[28]
            T_init = row.tolist()[29]
            i_enter = row.tolist()[30]
            i_exit = row.tolist()[31]
            T_ycl = row.tolist()[32]
            p_init = row.tolist()[33]
            delta_wall = row.tolist()[34]

    return (
        first_list,
        second_list,
        D_kp,
        mode,
        h,
        delta_ct,
        delta_p,
        delta_ct_HAP,
        beta,
        gamma,
        t_N_min,
        lymbda_material,
        k,
        Pr,
        alpha,
        T_ct_g,
        T_ct_o,
        mu_og,
        R_og,
        Cp_t_og,
        Cp_t_ct,
        phi,
        p_k,
        T_k,
        epsilon_h2o,
        epsilon_co2,
        epsilon_ct,
        m_oxl,
        coolant_type,
        T_init,
        i_enter,
        i_exit,
        T_ycl,
        p_init,
        delta_wall,
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
