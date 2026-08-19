# Battery preheat optimization prototype

This directory contains an SDK-independent prototype for the competition task.
It does not use `toolkit.7z`, and the included route is explicitly synthetic.

## What is implemented

- Bilinear interpolation for the provided `R_int`, `U_oc`, and `P_charge` maps.
- Coupled battery electrical, thermal, and SOC simulation with a 0.5 s step.
- PTC control that switches on once and remains at 6 kW until arrival.
- Exhaustive 1 s search over all possible preheat start times.
- Hard filtering for terminal temperature `[20, 25] C` and terminal SOC `>= 10%`.
- The published 30-point charging-time and 20-point heating-energy normalization.
- CSV result export and all five required PNG curves.
- A `candidate_sweep.csv` audit table containing every feasible strategy and its score.

## Run

From PowerShell:

```powershell
./run_prototype.ps1
```

The default input is `example_route_synthetic.csv`, which exists only to test the
software. Never report its optimum as an official competition result.

To use another route:

```powershell
./run_prototype.ps1 -Route path/to/route.csv -InitialTemperatureC 0 -InitialSocPercent 65
```

Route CSV columns:

```text
s_km,v_kmh,p_drive_kw,t_env_c
```

## Official SDK integration checklist

When the competition SDK arrives:

1. Move the map interpolation, simulation, and optimizer functions into the
   marked algorithm region of `student_solution.c`.
2. Replace CSV loading with `VehPwrPred_getPwrPred()` and populate `NavSeg`.
3. Use the SDK-provided initial temperature and SOC values.
4. Return SOC through the notification API in percent, even if internal state is
   represented as a fraction.
5. Map the four notification outputs and retain `chrg_time_s` for terminal output.
6. Re-run the official Mock verifier before accepting any result.

## Unit conventions

- Internal SOC in this prototype is stored as percent because all MAP axes and
  official output interfaces use percent.
- Electrical power is converted to watts before solving the Rint equation.
- Positive internal current means battery discharge. The SDK-reported current
  uses the opposite sign convention and must not be mixed into the forward model.
- Heating energy is electrical PTC input energy, in kWh. Thermal input uses the
  provided 0.95 conversion efficiency.
