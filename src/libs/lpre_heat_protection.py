from typing import List
from collections import deque
from math import floor
from numpy import round, pi, sqrt, radians, sin, cos, tan

# from scipy.interpolate import interp1d


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

    # How to find xs in correct meaning (via interp):
    # r_from_d = [i / 2 for i in d_list]
    # y_interp = interp1d(x_coord_list, r_from_d, kind="linear")
    # # xnew = arange(x_coord_list[1], x_coord_list[-1], x_coord_list[-1] / 32)
    # test = [(x_coord_list[i+1] - x_coord_list[i]) for i, el in enumerate(x_coord_list[1:])]
    # print(test)
    # ynew = y_interp(test)
    # print(ynew)

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

    Delta_xs_list = deque(Delta_xs_list)
    Delta_xs_list.appendleft("-")
    Delta_xs_list = list(Delta_xs_list)
    Delta_xs_list.append("-")

    Delta_S = [
        round(0.5 * pi * (d_list[i] + d_list[i + 1]) * Delta_xs_list[i], 6)
        for i, _ in enumerate(Delta_xs_list[1:])
    ]

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


def cooling_path_params(d_list: List[float]) -> List[List[float]]:
    # TODO -> 1) get init data (h, beta, gamma and etc)
    #         2) n_p_list calc
    #         3) n_p bad res
    #         4) FINAL RESULT MAY BE INCORRECT -> IT NEEDS RESEARCHES

    mode = "rightangle"
    h = 3
    delta_ct = 1
    delta_p = 1
    delta_ct_HAP = 3
    beta = radians(15)
    gamma = radians(90)
    h_p = (h - delta_p) / sin(gamma)

    t_N_min = 2.5

    d_avg_list = [
        diam * (1 + (2 * delta_ct + h) / diam) for _, diam in enumerate(d_list)
    ]
    d_avg_min = min(d_avg_list)
    d_kp = min(d_list)

    n_p_kp = floor(pi * d_avg_min * cos(beta) / t_N_min)

    n_p_min = round(n_p_kp / 2) if n_p_kp % 2 != 0 else n_p_kp
    t_N_min = pi * d_avg_min * cos(beta) / n_p_min

    n_p_list = []
    for _, el in enumerate(d_list):
        # power_of = 0
        # if el / d_min < pow(2, stepen):
        #     n_p_kp * pow(2, stepen)
        # else:
        #     stepen += 1
        if round(el / d_kp, 2) < 2:
            n_p_list.append(n_p_min)
        elif round(el / d_kp, 2) >= 2 and round(el / d_kp, 2) < 4:
            n_p_list.append(n_p_min * 2)
        elif round(el / d_kp, 2) >= 4 and round(el / d_kp, 2) < 8:
            n_p_list.append(n_p_min * 4)
        elif round(el / d_kp, 2) >= 8 and round(el / d_kp, 2) < 16:
            n_p_list.append(n_p_min * 8)
        elif round(el / d_kp, 2) >= 16 and round(el / d_kp, 2) < 32:
            n_p_list.append(n_p_min * 16)

    t_list = [pi * diam_avg / n_p_list[i] for i, diam_avg in enumerate(d_avg_list)]
    t_N_list = [t_val * cos(beta) for _, t_val in enumerate(t_list)]

    if mode == "shelevoi":
        f_list = [pi * diam_avg * h for _, diam_avg in enumerate(d_avg_list)]
        d_g_list = [2 * h for _ in d_avg_list]
        b_list = ["-" for _ in d_avg_list]
    elif mode == "rightangle":
        f_list = [
            t_N * h * (1 - delta_p / t_N) * n_p_list[i]
            for i, t_N in enumerate(t_N_list)
        ]
        d_g_list = [
            2 * h * (t_N - delta_p) / (t_N - delta_p + h)
            for _, t_N in enumerate(t_N_list)
        ]
        b_list = ["-" for _ in d_avg_list]
    else:
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