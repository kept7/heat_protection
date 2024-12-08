from typing import List, Type
from collections import deque
from numpy import pi, round, floor, sqrt, sin, cos, tan, tanh, log10
from thermo import (
    Chemical,
    ViscosityGas,
    ChemicalConstantsPackage,
    PRMIX,
    CEOSLiquid,
    CEOSGas,
    FlashPureVLS,
)
from scipy.interpolate import InterpolatedUnivariateSpline


def hello_world() -> None:
    print("hello world")


def chamber_params(
    x_coord_list: List[float], d_list: List[float], D_kp: float, F_kp: float
) -> List[List[int]]:

    D_ratio_list = [round(diam / D_kp, 6) for diam in d_list]

    F_list = [round((pow(F, 2) * pi) / 4, 6) for F in d_list]

    F_ratio_list = [round(F_rat / F_kp, 6) for F_rat in F_list]

    Delta_x_list = [
        round(x_coord_list[i + 1] - x_coord_list[i], 6)
        for i in range(len(x_coord_list) - 1)
    ]
    Delta_x_list.append("-")

    r_from_d = [diam / 2 for diam in d_list]

    Delta_xs_list = [
        round(
            sqrt(
                pow(x_coord_list[i + 1] - x_coord_list[i], 2)
                + pow(r_from_d[i + 1] - r_from_d[i], 2)
            ),
            6,
        )
        for i in range(len(x_coord_list) - 1)
    ]
    Delta_xs_list.append("-")

    Delta_S = [
        round(0.5 * pi * (d_list[i] + d_list[i + 1]) * Delta_xs_list[i], 6)
        for i in range(len(x_coord_list) - 1)
    ]
    Delta_S.append("-")

    result = [
        [
            x_coord_list[i],
            d_list[i],
            D_ratio_list[i],
            F_list[i],
            F_ratio_list[i],
            Delta_x_list[i],
            Delta_xs_list[i],
            Delta_S[i],
        ]
        for i, _ in enumerate(x_coord_list)
    ]

    return result, Delta_S, Delta_xs_list


def cooling_path_params(
    d_list: List[float],
    mode: int,
    h: float,
    delta_ct: float,
    delta_p: float,
    delta_ct_HAP: float,
    beta: float,
    gamma: float,
    t_N_min: float,
) -> List[List[float]]:
    # TODO ->
    #         2) decrease complexity of n_p_list calc

    h_p = (h - delta_p) / sin(gamma)

    d_avg_list = [
        diam * (1 + (2 * delta_ct + h) / diam) for _, diam in enumerate(d_list)
    ]
    d_avg_min = min(d_avg_list)

    n_p_kp = floor(pi * d_avg_min * cos(beta) / t_N_min)

    n_p_min = round(n_p_kp / 2) if n_p_kp % 2 != 0 else n_p_kp
    t_N_min = pi * d_avg_min * cos(beta) / n_p_min

    n_p_list = [n_p_min for _ in d_list]

    t_list = []
    t_N_list = []

    for i, el in enumerate(n_p_list):
        n_p_val = el
        t_val = pi * d_avg_list[i] / n_p_val
        t_N = t_val * cos(beta)

        while t_N > 0.007:
            n_p_val *= 2
            t_val = pi * d_avg_list[i] / n_p_val
            t_N = t_val * cos(beta)

        if n_p_val != el:
            n_p_list[i] = n_p_val

        t_list.append(round(t_val, 4))
        t_N_list.append(round(t_N, 4))

    if mode == 1:
        f_list = [pi * diam_avg * h for _, diam_avg in enumerate(d_avg_list)]
        d_g_list = [2 * h for _ in d_avg_list]
        b_list = ["-" for _ in d_avg_list]
    elif mode == 2:
        f_list = [
            t_N * h * (1 - delta_p / t_N) * n_p_list[i]
            for i, t_N in enumerate(t_N_list)
        ]
        d_g_list = [
            2 * h * (t_N - delta_p) / (t_N - delta_p + h)
            for _, t_N in enumerate(t_N_list)
        ]
        b_list = ["-" for _ in d_avg_list]
    elif mode == 3:
        b_list = [t_N - h / tan(gamma) for _, t_N in enumerate(t_N_list)]
        f_list = [
            n_p_list[i]
            * (
                t_N * h
                - delta_p
                * (sqrt(pow(h - delta_p, 2) + pow(t_N - b_list[i], 2)) + b_list[i])
            )
            for i, t_N in enumerate(t_N_list)
        ]
        d_g_list = [
            2
            * h
            * (
                (
                    t_N
                    - (delta_p / h)
                    * (sqrt(pow(h - delta_p, 2) + pow(t_N - b_list[i], 2)) + b_list[i])
                )
                / (t_N + sqrt(pow(h - delta_p, 2) + pow(t_N - b_list[i], 2)))
            )
            for i, t_N in enumerate(t_N_list)
        ]

    result = [
        [n_p_list[i], t_list[i], t_N_list[i], f_list[i], d_g_list[i], b_list[i]]
        for i, _ in enumerate(n_p_list)
    ]

    return result, t_N_list, f_list, d_g_list, h_p


def heat_flows_calc(
    x_coord_list: List[float],
    d_list: List[float],
    index_kp: int,
    k: float,
    Cp_t_og: float,
    Cp_t_ct: float,
    T_ct_g: float,
    T_ct_o: float,
    mu_og: float,
    R_og: float,
    Pr: float,
    p_k: float,
    T_k: float,
    phi: float,
    epsilon_h2o: float,
    epsilon_co2: float,
    epsilon_ct: float,
) -> List[List[float]]:

    epsilon = 1
    c_o = 5.67

    alpha_OTH = (
        1.813 * pow((2 / (k + 1)), (0.85 / (k - 1))) * pow((2 * k / (k + 1)), 0.425)
    )

    lymbda_list = []
    for i, d in enumerate(d_list):
        if x_coord_list[i] < x_coord_list[index_kp]:
            key = "Subsonic"
        else:
            key = "Supersonic"
        lymbda_list.append(
            round(get_lambda(pow((d / d_list[index_kp]), -2), k, key), 6)
        )

    beta_list = [
        lymbda * sqrt((k - 1) / (k + 1)) for _, lymbda in enumerate(lymbda_list)
    ]

    T_ct_OTH = [T_ct_g / T_ct_o for _, _ in enumerate(beta_list)]

    z_OTH_list = [
        pow(
            1.769
            * (
                (
                    1
                    - pow(beta, 2)
                    + pow(beta, 2)
                    * (
                        1
                        - 0.086
                        * ((1 - pow(beta, 2)) / (1 - T_ct_OTH[i] - 0.1 * pow(beta, 2)))
                    )
                )
                / (1 - T_ct_OTH[i] - 0.1 * pow(beta, 2))
            ),
            0.54,
        )
        for i, beta in enumerate(beta_list)
    ]

    B_list = [
        0.4842 * alpha_OTH * 0.01352 * pow(Z, 0.075) for _, Z in enumerate(z_OTH_list)
    ]

    C_p_cp_list = [0.5 * (Cp_t_og + Cp_t_ct) for _, _ in enumerate(x_coord_list)]

    S_list = [
        (2.065 * C_p_cp * (T_ct_o - T_ct_g) * pow(mu_og, 0.15))
        / (
            pow(R_og * T_ct_o, 0.425)
            * pow(1 + T_ct_OTH[i], 0.595)
            * pow(3 + T_ct_OTH[i], 0.15)
        )
        for i, C_p_cp in enumerate(C_p_cp_list)
    ]

    q_k_list = [
        round(
            (
                B_list[i]
                * (
                    ((1 - pow(beta_list[i], 2)) * epsilon * pow(p_k * 1000000, 0.85))
                    / (
                        pow(d_list[i] / d_list[index_kp], 1.82)
                        * pow(d_list[index_kp], 0.15)
                    )
                )
                * (S_list[i] / pow(Pr, 0.58))
            )
            / 1000000,  # -> MWt
            6,
        )
        for i, _ in enumerate(d_list)
    ]

    epsilon_g = epsilon_h2o + epsilon_co2 - epsilon_h2o * epsilon_co2
    epsilon_st_ef = (epsilon_ct + 1) / 2
    q_l_km = epsilon_st_ef * epsilon_g * c_o * pow(T_k / 100, 4)

    q_l_kc = phi * q_l_km / 1000000  # -> MWt

    q_l_list = []
    for i, x_coord in enumerate(x_coord_list):
        if x_coord <= 0.05:
            q_l_list.append(round(0.25 * q_l_kc, 6))
        elif (
            x_coord > 0.05
            and x_coord < x_coord_list[index_kp]
            and d_list[i] / d_list[index_kp] >= 1.2
        ):
            q_l_list.append(round(q_l_kc, 6))
        elif (
            x_coord > 0.05
            and x_coord < x_coord_list[index_kp]
            and d_list[i] / d_list[index_kp] < 1.2
        ):
            q_l_list.append(
                round(
                    q_l_kc * (1 - 12.5 * pow((1.2 - d_list[i] / d_list[index_kp]), 2)),
                    6,
                )
            )
        elif x_coord == x_coord_list[index_kp]:
            q_l_list.append(round(0.5 * q_l_kc, 6))
        elif x_coord > x_coord_list[index_kp]:
            q_l_list.append(
                round(0.5 * q_l_kc / pow(d_list[i] / d_list[index_kp], 2), 6)
            )

    q_sum_list = [q_k + q_l_list[i] for i, q_k in enumerate(q_k_list)]

    result = [
        [
            el,
            lymbda_list[i],
            beta_list[i],
            S_list[i],
            T_ct_OTH[i],
            q_k_list[i],
            q_l_list[i],
            q_sum_list[i],
        ]
        for i, el in enumerate(x_coord_list)
    ]

    return result, S_list, q_l_list, q_k_list, q_sum_list


def heat_in_cooling_path_calc(
    q_sum_list: List[float],
    Delta_S: List[float],
    f_list: List[float],
    d_g_list: List[float],
    t_list: List[float],
    i_enter: int,
    i_exit: int,
    T_init: int,
    m_oxl: float,
    coolant_type: str,
    delta_p: float,
    h_p: float,
    lymbda_material: float,
    beta: int,
    p_init: int,
) -> List[List[float]]:
    T_oxl_list = [T_init]
    # ecli bydet oshibka, to ckopee vsego ona tyt
    delta_cp_list = ["-"]

    for i in reversed(range(i_exit - 1, i_enter - 1)):
        j = 0
        delta_T_1 = 26
        delta_cp = 5.01
        T_oxl = T_oxl_list[j]
        while True:
            T_cp_1 = T_oxl + 0.5 * delta_T_1
            cp_cp_1 = find_thermo_phisics_coeffs(coolant_type, p_init, T_cp_1)[0]

            delta_T_2 = (
                0.5
                * (q_sum_list[i] + q_sum_list[i - 1])
                * 1000000
                * Delta_S[i]
                / (m_oxl * cp_cp_1)
            )

            T_cp_2 = T_oxl + 0.5 * delta_T_2
            cp_cp_2 = find_thermo_phisics_coeffs(coolant_type, p_init, T_cp_2)[0]

            delta_cp = ((cp_cp_2 - cp_cp_1) / cp_cp_1) * 100

            if delta_cp > 5 or delta_cp < 0:
                delta_T_1 = delta_T_2
            else:
                T_oxl_list = deque(T_oxl_list)
                T_oxl_list.appendleft(round(T_oxl + delta_T_2, 6))
                T_oxl_list = list(T_oxl_list)

                delta_cp_list = deque(delta_cp_list)
                delta_cp_list.appendleft(round(delta_cp, 6))
                delta_cp_list = list(delta_cp_list)

                j += 1
                break

    cp_oxl_list = [
        round(find_thermo_phisics_coeffs(coolant_type, p_init, T_oxl)[0], 6)
        for _, T_oxl in enumerate(T_oxl_list)
    ]

    mu_oxl_list = [
        round(find_thermo_phisics_coeffs(coolant_type, p_init, T_oxl)[1], 6)
        for _, T_oxl in enumerate(T_oxl_list)
    ]

    lymbda_oxl_list = [
        round(find_thermo_phisics_coeffs(coolant_type, p_init, T_oxl)[2], 6)
        for _, T_oxl in enumerate(T_oxl_list)
    ]

    K_oxl_list = [
        round(pow(lymbda_oxl_list[i], 0.6) * pow(cp_oxl / mu_oxl_list[i], 0.4), 6)
        for i, cp_oxl in enumerate(cp_oxl_list)
    ]

    rho_oxl_list = [
        p_init
        / ((8.314 / find_thermo_phisics_coeffs(coolant_type, p_init, T_oxl)[3]) * T_oxl)
        for _, T_oxl in enumerate(T_oxl_list)
    ]

    U_oxl_list = [
        round(m_oxl / (f_list[i] * rho_oxl), 6)
        for i, rho_oxl in enumerate(rho_oxl_list)
    ]

    alpha_oxl_list = [
        round(
            (0.023 * K_oxl * pow(U_oxl_list[i] * rho_oxl_list[i], 0.8))
            / pow(d_g_list[i], 0.2),
            6,
        )
        for i, K_oxl in enumerate(K_oxl_list)
    ]

    bi_list = [
        alpha_oxl * delta_p / lymbda_material
        for _, alpha_oxl in enumerate(alpha_oxl_list)
    ]

    psi_list = [h_p / delta_p * sqrt(2 * bi) for _, bi in enumerate(bi_list)]

    E_list = [round(tanh(psi) / psi, 6) for _, psi in enumerate(psi_list)]

    zeta_p = 1
    eta_p_list = [
        round(
            1
            + 1
            / cos(beta)
            * (2 * h_p / t_list[i] * E_val * zeta_p - delta_p / t_list[i]),
            6,
        )
        for i, E_val in enumerate(E_list)
    ]

    result = [
        [
            T_oxl,
            delta_cp_list[i],
            cp_oxl_list[i],
            lymbda_oxl_list[i],
            mu_oxl_list[i],
            K_oxl_list[i],
            U_oxl_list[i],
            alpha_oxl_list[i],
            E_list[i],
            eta_p_list[i],
        ]
        for i, T_oxl in enumerate(T_oxl_list)
    ]

    return (
        result,
        T_oxl_list,
        cp_oxl_list,
        alpha_oxl_list,
        eta_p_list,
        rho_oxl_list,
        U_oxl_list,
        mu_oxl_list,
    )


def find_thermo_phisics_coeffs(fluid: str, pressure: int, temperature: float) -> float:
    cas_number = 0
    chem_formula = 0

    if fluid == "UDMH" or fluid == "udmh":
        chem_formula = 1
    elif fluid == "Aerozine-50" or fluid == "aerozine":
        chem_formula = 1
    elif fluid == "kerosene T-1" or fluid == "T-1":
        chem_formula = 1
    elif fluid == "Ethanol" or fluid == "ethanol":
        cas_number = "64-17-5"
    elif fluid == "Water" or fluid == "water":
        cas_number = "7732-18-5"
    elif fluid == "Helium" or fluid == "helium":
        cas_number = "7440-59-7"
    elif fluid == "Ammonia" or fluid == "ammonia":
        cas_number = "7664-41-7"
    elif fluid == "Oxygen" or fluid == "oxygen":
        cas_number = "7782-44-7"
    elif fluid == "Hydrogen" or fluid == "hydrogen":
        cas_number = "1333-74-0"
    elif fluid == "Methane" or fluid == "methane":
        cas_number = "74-82-8"
    else:
        exit(1)

    if cas_number:
        specific_heat, viscosity, thermal_conductivity, molecular_weight = (
            thermo_lib_find(fluid, cas_number, temperature, pressure)
        )
        return specific_heat, viscosity, thermal_conductivity, molecular_weight
    elif chem_formula:
        specific_heat, viscosity, thermal_conductivity, molecular_weight = (
            rocketprops_lib_find(fluid, temperature)
        )
        return specific_heat, viscosity, thermal_conductivity, molecular_weight


def rocketprops_lib_find(fuel_name: str, temperature: float) -> float:
    if fuel_name == "UDMH" or fuel_name == "udmh":
        Cp_at_T_func, mu_at_T_func, lymbda_at_T_func = udmh_phys_func()
        molecular_weight = 0.0601
    elif fuel_name == "Aerozine-50" or fuel_name == "aerozine":
        Cp_at_T_func, mu_at_T_func, lymbda_at_T_func = aerozine_phys_func()
        molecular_weight = 0.09214
    elif fuel_name == "kerosene T-1" or fuel_name == "T-1":
        Cp_at_T_func, mu_at_T_func, lymbda_at_T_func = kerosene_phys_func()
        molecular_weight = 0.09981

    specific_heat = Cp_at_T_func(temperature)
    viscosity = mu_at_T_func(temperature)
    thermal_conductivity = lymbda_at_T_func(temperature)

    return specific_heat, viscosity, thermal_conductivity, molecular_weight


def udmh_phys_func() -> Type[InterpolatedUnivariateSpline]:
    temperature_init = [
        223,
        233,
        243,
        253,
        263,
        273,
        283,
        293,
        303,
        313,
        333,
        353,
        373,
        393,
        413,
        433,
        443,
        453,
        473,
    ]
    specific_heat_init = [
        2680,
        2700,
        2750,
        2760,
        2775,
        2795,
        2800,
        2810,
        2820,
        2830,
        2920,
        3000,
        3080,
        3160,
        3290,
        3470,
        3600,
        3730,
        4220,
    ]
    viscosity_init = [
        0.00253,
        0.00227,
        0.00162,
        0.0012,
        0.001,
        0.00075,
        0.00064,
        0.00051,
        0.00048,
        0.00041,
        0.00033,
        0.00031,
        0.00029,
        0.00024,
        0.00021,
        0.00018,
        0.00015,
        0.00014,
        0.00013,
    ]
    thermal_conductivity_init = [
        0.204,
        0.201,
        0.198,
        0.195,
        0.192,
        0.189,
        0.186,
        0.182,
        0.180,
        0.178,
        0.176,
        0.174,
        0.172,
        0.170,
        0.167,
        0.165,
        0.162,
        0.159,
        0.156,
    ]

    Cp_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, specific_heat_init, k=1
    )
    mu_at_T_func = InterpolatedUnivariateSpline(temperature_init, viscosity_init, k=1)
    lymbda_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, thermal_conductivity_init, k=1
    )

    return Cp_at_T_func, mu_at_T_func, lymbda_at_T_func


def aerozine_phys_func() -> Type[InterpolatedUnivariateSpline]:
    temperature_init = [
        263,
        268,
        273,
        278,
        283,
        288,
        293,
        298,
        303,
        308,
        313,
        318,
        323,
        328,
        333,
        338,
        343,
    ]
    specific_heat_init = [
        2850,
        2856,
        2870,
        2875,
        2887,
        2856,
        2902,
        2913,
        2922,
        2931,
        2940,
        2950,
        2959,
        2975,
        2970,
        2988,
        3000,
    ]
    viscosity_init = [
        0.00165,
        0.001475,
        0.001313,
        0.001175,
        0.001050,
        0.000956,
        0.000875,
        0.0008,
        0.000750,
        0.0007,
        0.00065,
        0.000613,
        0.000575,
        0.000537,
        0.0005,
        0.000458,
        0.00045,
    ]
    thermal_conductivity_init = [0.254 for _ in temperature_init]

    Cp_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, specific_heat_init, k=1
    )
    mu_at_T_func = InterpolatedUnivariateSpline(temperature_init, viscosity_init, k=1)
    lymbda_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, thermal_conductivity_init, k=1
    )

    return Cp_at_T_func, mu_at_T_func, lymbda_at_T_func


def kerosene_phys_func() -> Type[InterpolatedUnivariateSpline]:
    temperature_init = [
        223,
        233,
        253,
        273,
        293,
        313,
        333,
        353,
        373,
        393,
        413,
        433,
        453,
        473,
        533,
    ]
    specific_heat_init = [
        1890,
        1900,
        1950,
        2000,
        2050,
        2100,
        2200,
        2280,
        2360,
        2440,
        2550,
        2660,
        2800,
        2900,
        3060,
    ]
    viscosity_init = [
        0.012,
        0.0073,
        0.0035,
        0.002,
        0.0015,
        0.001,
        0.0008,
        0.0006,
        0.0005,
        0.00045,
        0.00039,
        0.00035,
        0.0003,
        0.00026,
        0.0002,
    ]
    thermal_conductivity_init = [
        0.127,
        0.125,
        0.123,
        0.120,
        0.117,
        0.114,
        0.11,
        0.108,
        0.104,
        0.102,
        0.099,
        0.096,
        0.093,
        0.09,
        0.084,
    ]

    Cp_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, specific_heat_init, k=1
    )
    mu_at_T_func = InterpolatedUnivariateSpline(temperature_init, viscosity_init, k=1)
    lymbda_at_T_func = InterpolatedUnivariateSpline(
        temperature_init, thermal_conductivity_init, k=1
    )

    return Cp_at_T_func, mu_at_T_func, lymbda_at_T_func


def thermo_lib_find(
    fluid: str, cas_number: str, temperature: float, pressure: int
) -> float:
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    liquid = CEOSLiquid(
        PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs
    )
    gas = CEOSGas(
        PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs
    )
    flasher = FlashPureVLS(
        constants, correlations, gas=gas, liquids=[liquid], solids=[]
    )

    res = flasher.flash(T=temperature, P=pressure)
    specific_heat = res.Cp_mass()
    molecular_weight = res.MW() / 1000

    fluid_res = Chemical(cas_number, T=temperature, P=pressure)
    thermal_conductivity = fluid_res.k

    fluid_viscosity = ViscosityGas(CASRN=cas_number)
    viscosity = fluid_viscosity.calculate_P(
        T=temperature, P=pressure, method="COOLPROP"
    )

    return specific_heat, viscosity, thermal_conductivity, molecular_weight


def temperature_approx(
    x_coord_list: List[float],
    q_l_list: List[float],
    q_k_list: List[float],
    T_oxl_list: List[float],
    alpha_oxl_list: List[float],
    eta_p_list: List[float],
    T_ct_o: float,
    T_ycl: float,
    delta_ct: float,
    lymbda_material: float,
) -> List[float]:
    appr_T_ct_g_list = [
        round(
            (
                T_ct_o / (T_ct_o - T_ycl)
                + T_oxl_list[i]
                / (
                    (
                        delta_ct / lymbda_material
                        + 1 / (alpha_oxl_list[i] * eta_p_list[i])
                    )
                    * q_k_var
                    * 1000000
                )
                + q_l_list[i] / q_k_var
            )
            / (
                1 / (T_ct_o - T_ycl)
                + 1
                / (
                    (
                        delta_ct / lymbda_material
                        + 1 / (alpha_oxl_list[i] * eta_p_list[i])
                    )
                    * q_k_var
                    * 1000000
                )
            ),
            6,
        )
        for i, q_k_var in enumerate(q_k_list)
    ]

    result = [[x_coord, appr_T_ct_g_list[i]] for i, x_coord in enumerate(x_coord_list)]

    return result, appr_T_ct_g_list


def temperature_with_local_curtain():
    result1 = "first table"
    result2 = "second table"
    return result1, result2


def temperature_second_approx(
    S_list: List[float],
    q_k_list: List[float],
    cp_oxl_list: List[float],
    T_ct_g_list: List[float],
    mu_og: float,
    R_og: float,
    T_ct_o: float,
) -> List[float]:
    S_list_sec_approx_list = [
        (2.065 * Cp_oxl * (T_ct_o - T_ct_g_list[i]) * pow(mu_og, 0.15))
        / (
            pow(R_og * T_ct_o, 0.425)
            * pow(1 + (T_ct_g_list[i] / T_ct_o), 0.595)
            * pow(3 + (T_ct_g_list[i] / T_ct_o), 0.15)
        )
        for i, Cp_oxl in enumerate(cp_oxl_list)
    ]

    q_k_sec_approx_list = [
        q_k_var * S_list_sec_approx_list[i] / S_list[i]
        for i, q_k_var in enumerate(q_k_list)
    ]

    return q_k_sec_approx_list, S_list_sec_approx_list


def calculation_temperature_coolant_wall(
    second_appr_T_ct_g_list: List[float],
    q_sum_list_sec_approx: List[float],
    delta_ct: float,
    lymbda_material: float,
) -> List[float]:

    temperature_coolant_wall_list = [
        T_ct_g - (delta_ct / lymbda_material) * q_sum_list_sec_approx[i]
        for i, T_ct_g in enumerate(second_appr_T_ct_g_list)
    ]

    return temperature_coolant_wall_list


def calc_coolant_pressure_losses(
    Delta_xs_list: List[float],
    d_g_list: List[float],
    rho_oxl_sec_approx_list: List[float],
    U_oxl_sec_approx_list: List[float],
    mu_oxl_sec_approx_list: List[float],
    t_N_list: List[float],
    beta: float,
    delta_wall: float,
    delta_p: float,
    h_p: float,
) -> List[List[float]]:
    Re_num_list = [
        round(
            rho_oxl
            * U_oxl_sec_approx_list[i]
            * d_g_list[i]
            / mu_oxl_sec_approx_list[i],
            0,
        )
        for i, rho_oxl in enumerate(rho_oxl_sec_approx_list)
    ]

    omega_list = omega_relation(t_N_list, delta_p, h_p)

    delta_OTH_ct_list = [
        round(float(delta_wall) / float(d_g), 4) for _, d_g in enumerate(d_g_list)
    ]

    l_list = [
        round(Delta_xs_list[i] / cos(beta), 3) for i in range(len(Delta_xs_list) - 1)
    ]
    l_list.append("-")

    zeta_list = []
    for i, delta_OTH_ct in enumerate(delta_OTH_ct_list):
        zeta = "-"
        if Re_num_list[i] <= 3500 and Re_num_list[i] >= 0:
            zeta = 64 * omega_list[i] / Re_num_list[i]
        elif Re_num_list[i] >= 560 / delta_OTH_ct:
            zeta = omega_list[i] / pow(2 * log10(3.7 / delta_OTH_ct), 2)
        elif Re_num_list[i] > 3500 and Re_num_list[i] < 560 / delta_OTH_ct:
            if delta_OTH_ct >= 0.01 and delta_OTH_ct <= 0.6001:
                zeta = (
                    0.1
                    * pow(1.46 * delta_OTH_ct + 100 / Re_num_list[i], 0.25)
                    * omega_list[i]
                )
            elif delta_OTH_ct >= 0.0001 and delta_OTH_ct <= 0.01:
                zeta = (
                    1.42 * omega_list[i] / pow(log10(Re_num_list[i] / delta_OTH_ct), 2)
                )
        zeta_list.append(round(zeta, 5))

    delta_press = [
        round(
            zeta_list[i]
            * ((rho_oxl_sec_approx_list[i] * pow(U_oxl_sec_approx_list[i], 2)) / 2)
            * l_list[i]
            / d_g_list[i],
            0,
        )
        for i in range(len(rho_oxl_sec_approx_list) - 1)
    ]
    delta_press.append("-")

    Re_gr_list = [
        round(560 / delta_OTH_ct, 0) for _, delta_OTH_ct in enumerate(delta_OTH_ct_list)
    ]

    result = [
        [
            Re_num,
            delta_OTH_ct_list[i],
            Re_gr_list[i],
            zeta_list[i],
            l_list[i],
            delta_press[i],
        ]
        for i, Re_num in enumerate(Re_num_list)
    ]

    return result


def omega_relation(t_N_list: List[float], delta_p: float, h_p: float) -> List[float]:
    omega_init = [1.5, 1.32, 1.25, 1.1, 1.03, 0.97, 0.91, 0.9]
    a_b_init = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]

    omega_func = InterpolatedUnivariateSpline(a_b_init, omega_init, k=1)
    omega_list = [omega_func((t_N - delta_p) / h_p) for _, t_N in enumerate(t_N_list)]

    return omega_list


def get_lambda(q: float, k: float, key: str) -> float:

    if key == "Subsonic":
        lmbd = 0.3
    elif key == "Supersonic":
        lmbd = 1.01
    while True:
        if key == "Subsonic":
            f = (
                lmbd
                * (1 - (k - 1) / (k + 1) * lmbd**2) ** (1 / (k - 1))
                * ((k + 1) / 2) ** (1 / (k - 1))
                - q
            )
            delta_lamda = 0.01
            fd = (
                (lmbd + delta_lamda)
                * (1 - (k - 1) / (k + 1) * (lmbd + delta_lamda) ** 2) ** (1 / (k - 1))
                * ((k + 1) / 2) ** (1 / (k - 1))
            ) / delta_lamda
        elif key == "Supersonic":
            f = (
                lmbd
                * (1 - (k - 1) / (k + 1) * lmbd**2) ** (1 / (k - 1))
                * ((k + 1) / 2) ** (1 / (k - 1))
                - q
            )
            delta_lamda = 0.01
            fd = (
                (lmbd - delta_lamda)
                * (1 - (k - 1) / (k + 1) * (lmbd - delta_lamda) ** 2) ** (1 / (k - 1))
                * ((k + 1) / 2) ** (1 / (k - 1))
            ) / -delta_lamda

        lambda1 = lmbd - f / fd

        if abs((lambda1 - lmbd) / lambda1) < 0.000001:
            lambda1
            break
        else:
            lmbd = lambda1

    return lambda1