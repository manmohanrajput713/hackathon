from __future__ import annotations

from numpy.typing import NDArray

def q_primary(
    production: NDArray, time: NDArray, gain_producer: float, tau_producer: float
) -> NDArray: ...
def q_crm_perpair(injection: NDArray, time: NDArray, gains: NDArray, taus: NDArray) -> NDArray: ...
def q_bhp(pressure_local: NDArray, pressure: NDArray, v_matrix: NDArray) -> NDArray: ...
def calc_A_ij(x_i: float, y_i: float, x_j: float, y_j: float, y_d: float, m: NDArray) -> float: ...
def water_front_velocity(
    flow_rate: float,
    phi: float,
    flow_cross_section: float,
    viscosity_oil: float,
    viscosity_water: float,
    sat_oil: float,
    sat_water: float,
    sat_oil_r: float,
    sat_water_c: float,
    sat_gas_c: float,
    n_oil: float,
    n_water: float,
) -> float: ...
def breakthrough_sw(
    viscosity_oil: float,
    viscosity_water: float,
    sat_oil_r: float,
    sat_water_c: float,
    sat_gas_c: float,
    n_oil: float,
    n_water: float,
) -> float: ...
def fractional_flow_water(
    k_oil: float,
    k_water: float,
    viscosity_oil: float,
    viscosity_water: float,
) -> float: ...
def k_rel_oil(
    sat_oil: float,
    sat_oil_r: float,
    sat_water_c: float,
    sat_gas_c: float,
    n_oil: float,
) -> float: ...
def k_rel_water(
    sat_water: float,
    sat_oil_r: float,
    sat_water_c: float,
    sat_gas_c: float,
    n_water: float,
) -> float: ...
def w_ek(w_d: NDArray, delta_pressure: NDArray) -> NDArray: ...
