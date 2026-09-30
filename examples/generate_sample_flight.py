"""
Generate the example flight used in the README: a synthetic six-leg
lawnmower survey at 10 m/s with 1 Hz SPS30 logging, a 3 m/s along-track
wind, about 1 m of GPS jitter, two plumes, and two single-sample sensor
dropouts.

Usage (from the repository root):
    python examples/generate_sample_flight.py

Writes examples/sample_flight.csv. The random seed is fixed, so the output
is identical on every run.
"""

import csv
import os

import numpy as np

R_EARTH = 6_371_000.0                     # Earth radius [m]
LAT0, LON0 = 42.9849, -81.2453            # survey start point [deg]
ALT_AGL = 12.0                            # survey altitude [m AGL]
DT = 1.0                                  # SPS30 logging interval [s]
TAU = 2.0                                 # sensor time constant [s]
V_AIR = 10.0                              # cruise airspeed [m/s]
WIND = 3.0                                # along-track wind [m/s]
N_LEGS = 6                                # number of survey legs
LEG_S = 100.0                             # duration of one leg [s]
TURN_S = 12.0                             # duration of one turn [s]
GPS_SIGMA = 1.0                           # GPS fix-to-fix jitter [m]
SEED = 2026


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Flight path: alternating north and south legs joined by 180 deg turns.
    # Ground speed is airspeed plus the along-track wind, which reverses
    # sign on every leg.
    vg, hd = [], []
    for k in range(N_LEGS):
        h = 0.0 if k % 2 == 0 else 180.0
        s = 1 if k % 2 == 0 else -1
        for _ in range(int(LEG_S / DT)):
            vg.append(V_AIR + s * WIND + rng.normal(0, 0.08))
            hd.append(h)
        for h2 in np.linspace(h, h + 180.0, int(TURN_S / DT)) % 360.0:
            vg.append(7.5 + rng.normal(0, 0.1))
            hd.append(h2)
    vg, hd = np.array(vg), np.array(hd)
    n = vg.size
    t = np.arange(n) * DT

    # Integrate position, then add GPS jitter.
    north = np.zeros(n)
    east = np.zeros(n)
    for i in range(1, n):
        d = vg[i - 1] * DT
        north[i] = north[i - 1] + d * np.cos(np.radians(hd[i - 1]))
        east[i] = east[i - 1] + d * np.sin(np.radians(hd[i - 1]))
    north += rng.normal(0, GPS_SIGMA, n)
    east += rng.normal(0, GPS_SIGMA, n)
    lat = LAT0 + np.degrees(north / R_EARTH)
    lon = LON0 + np.degrees(east / (R_EARTH * np.cos(np.radians(LAT0))))
    alt = ALT_AGL + rng.normal(0, 0.3, n)

    # True concentration (baseline plus two plumes), smeared by the sensor
    # response and corrupted with measurement noise.
    x = (4.0 + 48.0 * np.exp(-0.5 * ((t - 55) / 5) ** 2)
         + 75.0 * np.exp(-0.5 * ((t - 390) / 6) ** 2))
    kern = np.exp(-np.arange(n) * DT / TAU)
    ii, jj = np.indices((n, n))
    df = ii - jj
    A = np.where(df >= 0, kern[np.clip(df, 0, n - 1)], 0.0)
    A /= A.sum(axis=1, keepdims=True)
    y = A @ x + rng.normal(0, 1.2, n)
    y[[140, 505]] = np.nan                # two single-sample dropouts

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sample_flight.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "lat", "lon", "alt", "raw_pm25",
                    "ground_speed"])
        for i in range(n):
            w.writerow([f"{t[i]:.1f}", f"{lat[i]:.8f}", f"{lon[i]:.8f}",
                        f"{alt[i]:.2f}",
                        "" if np.isnan(y[i]) else f"{y[i]:.3f}",
                        f"{vg[i]:.3f}"])
    print(f"wrote {out_path}: {n} rows")


if __name__ == "__main__":
    main()
