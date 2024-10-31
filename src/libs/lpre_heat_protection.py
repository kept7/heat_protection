from typing import List
from collections import deque
from numpy import pi, sqrt


def hello_world() -> None:
    print("hello world")


def chamber_params(
    x_coord_list: List[float], d_list: List[float], D_kp: float, F_kp: float
) -> List[List[int]]:

    D_ratio_list = [round(i / D_kp, 6) for i in d_list]

    F_list = [round((pow(i, 2) * pi) / 4, 6) for i in d_list]

    F_ratio_list = [round(i / F_kp, 6) for i in F_list]

    Delta_x_list = [
        round(el - x_coord_list[i], 6) for i, el in enumerate(x_coord_list[1:])
    ]
    Delta_x_list = deque(Delta_x_list)
    Delta_x_list.appendleft("-")
    Delta_x_list = list(Delta_x_list)

    r_from_d = [i / 2 for i in d_list]

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
