from typing import List
from collections import deque
from numpy import pi, round, floor, sqrt, radians, sin, cos, tan


def hello_world() -> None:
    print("hello world")


def chamber_params(
    x_coord_list: List[float], d_list: List[float], D_kp: float, F_kp: float
) -> List[List[int]]:

    D_ratio_list = [round(diam / D_kp, 6) for diam in d_list]

    F_list = [round((pow(F, 2) * pi) / 4, 6) for F in d_list]

    F_ratio_list = [round(F_rat / F_kp, 6) for F_rat in F_list]

    Delta_x_list = [
        round(x_coord - x_coord_list[i], 6)
        for i, x_coord in enumerate(x_coord_list[1:])
    ]
    Delta_x_list = deque(Delta_x_list)
    Delta_x_list.appendleft("-")
    Delta_x_list = list(Delta_x_list)

    r_from_d = [diam / 2 for diam in d_list]

    Delta_xs_list = [
        round(
            sqrt(
                pow(x_coord_list[i + 1] - x_coord_list[i], 2)
                + pow(r_from_d[i + 1] - r_from_d[i], 2)
            ),
            6,
        )
        for i, _ in enumerate(d_list[1:])
    ]

    Delta_S = [
        round(0.5 * pi * (d_list[i] + d_list[i + 1]) * Delta_xs_list[i], 6)
        for i, _ in enumerate(Delta_xs_list[1:])
    ]

    Delta_xs_list = deque(Delta_xs_list)
    Delta_xs_list.appendleft("-")
    Delta_xs_list = list(Delta_xs_list)
    Delta_xs_list.append("-")

    Delta_S = deque(Delta_S)
    Delta_S.appendleft("-")
    Delta_S = list(Delta_S)
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

    return result


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

    return result


def heat_flows_calc(
    x_coord_list: List[float],
    d_list: List[float],
    W_list: List[float],
    Cp_t_og_list: List[float],
    Cp_t_ct_list: List[float],
    mu_list: List[float],
    p_list: List[float],
    F_kc: float,
    F_kp: float,
    index_kp: int,
    k: float,
    T_ct_g: float,
    T_ct_o: float,
    mu_og: float,
    R_og: float,
    Pr: float,
    T_k: float,
) -> List[List[float]]:

    epsilon = 1 if F_kc / F_kp > 3.5 else 0.8  # value else 0.8?
    c_o = 5.67
    epsilon_ct = 0.8
    phi = 0.9

    alpha_OTH = (
        1.813 * pow((2 / (k + 1)), (0.85 / (k - 1))) * pow((2 * k / (k + 1)), 0.425)
    )

    W_kp = W_list[index_kp]

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

    lymbda_list = [W / W_kp for _, W in enumerate(W_list)]

    beta_list = [
        lymbda * sqrt((k - 1) / (k + 1)) for _, lymbda in enumerate(lymbda_list)
    ]

    B_list = [
        0.4842 * alpha_OTH * 0.01352 * pow(Z, 0.075) for _, Z in enumerate(z_OTH_list)
    ]

    C_p_cp_list = [
        0.5 * (Cp_t_og_list[i] + Cp_t_ct) for i, Cp_t_ct in enumerate(Cp_t_ct_list)
    ]

    S_list = [
        (2.065 * C_p_cp * (T_ct_o - T_ct_g) * pow(mu_og, 0.15))
        / (
            pow(R_og * T_ct_o, 0.425)
            * pow(1 + T_ct_OTH, 0.595)
            * pow(3 + T_ct_OTH, 0.15)
        )
        for _, C_p_cp in enumerate(C_p_cp_list)
    ]

    T_ct_OTH = [T_ct_g / T_ct_o for _, _ in enumerate(beta_list)]

    q_k_list = [
        B_list[i]
        * (
            ((1 - pow(beta_list[i], 2)) * epsilon * pow(p_list[0], 0.85))
            / (pow(d_list[i] / d_list[index_kp], 1.82) * pow(d_list[index_kp], 0.15))
        )
        * (S_list[i] / pow(Pr, 0.58))
        for i, _ in enumerate(d_list)
    ]

    epsilon_st_ef = (epsilon_ct + 1) / 2
    q_l_km = epsilon_st_ef * epsilon_g * c_o * pow(T_k / 100, 4)

    ro_list = [p_list[i] / ((8314 / mu) * T_k) for i, mu in enumerate(mu_list)]

    q_l_kc = phi * q_l_km

    q_l_list = [ for _, d in enumerate(d_list)]

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

    return result
