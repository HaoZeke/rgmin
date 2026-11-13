from typing import Callable

class Session:
    def __init__(
        self,
        kind: str,
        n: int,
        *,
        max_atom: float = 0.2,
        wolfe: bool = False,
        eon_cap: bool = False,
        memory: int = 8,
        dt: float = 0.1,
        dt_max: float = 1.0,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        alpha: float = 0.1,
        f_alpha: float = 0.99,
        downhill: bool = False,
    ) -> None: ...
    def step(
        self,
        x: list[float],
        callback: Callable[[list[float]], tuple[float, list[float]]],
    ) -> tuple[list[float], float, float]: ...

version: str
