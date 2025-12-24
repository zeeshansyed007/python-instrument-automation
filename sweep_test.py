from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) Instrument abstraction
# -----------------------------
class PowerSupply:
    """Abstract power supply interface."""

    def set_voltage(self, v: float) -> None:
        raise NotImplementedError

    def output_on(self) -> None:
        raise NotImplementedError

    def output_off(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DMM:
    """Abstract digital multimeter interface."""

    def measure_voltage(self) -> float:
        raise NotImplementedError

    def measure_current(self) -> float:
        raise NotImplementedError

    def close(self) -> None:
        pass


# -----------------------------
# 2) SIMULATED instruments (works without hardware)
# -----------------------------
@dataclass
class SimLoadModel:
    r_ohm: float = 10.0          # load resistance
    v_drop: float = 0.05         # wiring/reg drop
    noise_v: float = 0.003       # voltage noise std dev
    noise_i: float = 0.002       # current noise std dev


class SimPowerSupply(PowerSupply):
    def __init__(self) -> None:
        self._v_set = 0.0
        self._on = False

    @property
    def v_set(self) -> float:
        return self._v_set

    @property
    def is_on(self) -> bool:
        return self._on

    def set_voltage(self, v: float) -> None:
        self._v_set = float(v)

    def output_on(self) -> None:
        self._on = True

    def output_off(self) -> None:
        self._on = False


class SimDMM(DMM):
    def __init__(self, psu: SimPowerSupply, model: SimLoadModel) -> None:
        self.psu = psu
        self.model = model
        self.rng = np.random.default_rng(7)

    def _true_vout(self) -> float:
        if not self.psu.is_on:
            return 0.0
        v = max(self.psu.v_set - self.model.v_drop, 0.0)
        return v

    def measure_voltage(self) -> float:
        v = self._true_vout()
        return float(v + self.rng.normal(0.0, self.model.noise_v))

    def measure_current(self) -> float:
        v = self._true_vout()
        i = (v / self.model.r_ohm) if self.model.r_ohm > 0 else 0.0
        return float(i + self.rng.normal(0.0, self.model.noise_i))


# -----------------------------
# 3) REAL instruments (optional, later)
# -----------------------------
def try_open_real_instruments(psu_resource: str, dmm_resource: str) -> Tuple[PowerSupply, DMM]:
    """
    Optional real instrument support via PyVISA.
    This function will only be used if you run with --mode real.
    """
    import pyvisa  # installed already

    rm = pyvisa.ResourceManager()
    psu = rm.open_resource(psu_resource)
    dmm = rm.open_resource(dmm_resource)

    class VisaPSU(PowerSupply):
        def set_voltage(self, v: float) -> None:
            # Generic SCPI pattern (vendor models differ)
            psu.write(f"VOLT {v}")

        def output_on(self) -> None:
            psu.write("OUTP ON")

        def output_off(self) -> None:
            psu.write("OUTP OFF")

        def close(self) -> None:
            psu.close()

    class VisaDMM(DMM):
        def measure_voltage(self) -> float:
            # Generic SCPI measurement query (vendor models differ)
            return float(dmm.query("MEAS:VOLT:DC?"))

        def measure_current(self) -> float:
            return float(dmm.query("MEAS:CURR:DC?"))

        def close(self) -> None:
            dmm.close()

    return VisaPSU(), VisaDMM()


# -----------------------------
# 4) Test procedure
# -----------------------------
def run_sweep(
    psu: PowerSupply,
    dmm: DMM,
    v_start: float,
    v_stop: float,
    steps: int,
    settle_s: float,
) -> pd.DataFrame:
    voltages = np.linspace(v_start, v_stop, steps)
    records: List[dict] = []

    psu.output_on()
    time.sleep(0.2)

    for v in voltages:
        psu.set_voltage(float(v))
        time.sleep(settle_s)

        v_meas = dmm.measure_voltage()
        i_meas = dmm.measure_current()
        p_meas = v_meas * i_meas

        records.append({
            "v_set_v": float(v),
            "v_meas_v": float(v_meas),
            "i_meas_a": float(i_meas),
            "p_meas_w": float(p_meas),
        })

    psu.output_off()
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "samples": int(len(df)),
        "v_set_min": float(df["v_set_v"].min()),
        "v_set_max": float(df["v_set_v"].max()),
        "v_meas_mean": float(df["v_meas_v"].mean()),
        "i_meas_mean": float(df["i_meas_a"].mean()),
        "p_meas_mean": float(df["p_meas_w"].mean()),
    }


def plot_results(df: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(df["v_set_v"], df["v_meas_v"])
    plt.xlabel("Vset (V)")
    plt.ylabel("Vmeas (V)")
    plt.title("Voltage Sweep: Set vs Measured")
    plt.grid(True)
    plt.savefig("sweep_v_plot.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(df["v_set_v"], df["i_meas_a"])
    plt.xlabel("Vset (V)")
    plt.ylabel("Imeas (A)")
    plt.title("Voltage Sweep: Current vs Set Voltage")
    plt.grid(True)
    plt.savefig("sweep_i_plot.png", dpi=150)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Python instrument automation sweep (SIM/REAL).")
    parser.add_argument("--mode", choices=["sim", "real"], default="sim", help="Run with simulated or real instruments.")
    parser.add_argument("--v-start", type=float, default=0.0)
    parser.add_argument("--v-stop", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--settle-s", type=float, default=0.2)
    parser.add_argument("--psu", type=str, default="", help="VISA resource string for PSU (real mode).")
    parser.add_argument("--dmm", type=str, default="", help="VISA resource string for DMM (real mode).")
    args = parser.parse_args()

    if args.mode == "sim":
        psu = SimPowerSupply()
        dmm = SimDMM(psu, SimLoadModel(r_ohm=10.0))
    else:
        if not args.psu or not args.dmm:
            raise SystemExit("Real mode requires --psu and --dmm VISA resource strings.")
        psu, dmm = try_open_real_instruments(args.psu, args.dmm)

    try:
        df = run_sweep(psu, dmm, args.v_start, args.v_stop, args.steps, args.settle_s)
        df.to_csv("sweep_results.csv", index=False)

        rep = summarize(df)
        pd.DataFrame([rep]).to_csv("sweep_summary.csv", index=False)

        print("=== Sweep Summary ===")
        for k, v in rep.items():
            print(f"{k}: {v}")

        plot_results(df)

        print("\nSaved files:")
        print("- sweep_results.csv")
        print("- sweep_summary.csv")
        print("- sweep_v_plot.png")
        print("- sweep_i_plot.png")
    finally:
        psu.close()
        dmm.close()


if __name__ == "__main__":
    main()
