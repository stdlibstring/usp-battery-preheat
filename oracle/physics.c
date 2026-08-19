/*
 * physics.c — Shared physics library for ML/RL environments.
 *
 * Thin C API over preheat_model.h (same arithmetic as student_solution.c
 * and the oracle generator). Built as physics.dll and loaded from Python
 * via ctypes so that RL training, evaluation and the offline oracle all
 * use ONE physics implementation.
 *
 * Build:
 *   gcc -Wall -Wextra -O2 -shared -o physics.dll physics.c -lm
 */
#include "preheat_model.h"

#include <stdlib.h>
#include <string.h>

/* segs layout: 4 doubles per segment [s_km, v_kmh, P_drive_kW, T_env_C].
 *
 * out (6 doubles):
 *   [0] T_end (C)
 *   [1] SOC_end (%)
 *   [2] E_heat (kWh)
 *   [3] charge_time_s
 *   [4] charge_power_kW
 *   [5] remaining distance at t_start (km)
 * Returns 1 on success, 0 on failure.
 */
int ph_dll_simulate(const double *segs, int n_segs,
                    double t0_c, double soc0_pct,
                    double dt, double t_start, double *out)
{
    PhSeg s[PH_MAX_SEGS];
    if (segs == NULL || out == NULL || n_segs <= 0 || n_segs > PH_MAX_SEGS) {
        return 0;
    }
    for (int i = 0; i < n_segs; ++i) {
        s[i].s_km       = segs[i * 4 + 0];
        s[i].v_kmh      = segs[i * 4 + 1];
        s[i].p_drive_kw = segs[i * 4 + 2];
        s[i].t_env_c    = segs[i * 4 + 3];
    }
    PhRoute r;
    if (!ph_route_build(&r, s, n_segs)) return 0;

    const int need = (int)(r.total_time_s / dt) + r.n * 4 + 64;
    /* single-shot call: allocate prefix on the stack-heap boundary */
    PhSnap *snaps = (PhSnap *)malloc((size_t)need * sizeof(PhSnap));
    if (snaps == NULL) return 0;

    PhPrefix pf;
    pf.snaps = snaps;
    pf.n = 0;
    if (!ph_prefix_build(&r, t0_c, soc0_pct, dt, snaps, need, &pf.n,
                         &pf.end_temp_c, &pf.end_soc_pct)) {
        free(snaps);
        return 0;
    }

    const PhSimResult sim = ph_simulate_candidate(&r, &pf, dt, t_start);
    free(snaps);
    if (!sim.valid) return 0;

    out[0] = sim.end_temp_c;
    out[1] = sim.end_soc_pct;
    out[2] = sim.heat_kwh;
    out[3] = sim.charge_time_s;
    out[4] = sim.charge_power_kw;
    out[5] = ph_remaining_km_at_time(&r, t_start);
    return 1;
}

/* Batch variant: build the prefix ONCE, then simulate many candidate
 * start times (used by the online safety layer and PPO rollouts).
 * out must hold 6 * n_starts doubles, same layout as ph_dll_simulate.
 * Returns the number of successful simulations. */
int ph_dll_simulate_many(const double *segs, int n_segs,
                          double t0_c, double soc0_pct,
                          double dt, const double *t_starts, int n_starts,
                          double *out)
{
    PhSeg s[PH_MAX_SEGS];
    if (segs == NULL || out == NULL || t_starts == NULL ||
        n_segs <= 0 || n_segs > PH_MAX_SEGS || n_starts <= 0) {
        return 0;
    }
    for (int i = 0; i < n_segs; ++i) {
        s[i].s_km       = segs[i * 4 + 0];
        s[i].v_kmh      = segs[i * 4 + 1];
        s[i].p_drive_kw = segs[i * 4 + 2];
        s[i].t_env_c    = segs[i * 4 + 3];
    }
    PhRoute r;
    if (!ph_route_build(&r, s, n_segs)) return 0;

    const int need = (int)(r.total_time_s / dt) + r.n * 4 + 64;
    PhSnap *snaps = (PhSnap *)malloc((size_t)need * sizeof(PhSnap));
    if (snaps == NULL) return 0;

    PhPrefix pf;
    pf.snaps = snaps;
    pf.n = 0;
    if (!ph_prefix_build(&r, t0_c, soc0_pct, dt, snaps, need, &pf.n,
                         &pf.end_temp_c, &pf.end_soc_pct)) {
        free(snaps);
        return 0;
    }

    int ok_count = 0;
    for (int k = 0; k < n_starts; ++k) {
        const PhSimResult sim = ph_simulate_candidate(&r, &pf, dt, t_starts[k]);
        double *o = out + (size_t)k * 6;
        if (!sim.valid) {
            for (int j = 0; j < 6; ++j) o[j] = 0.0;
            continue;
        }
        o[0] = sim.end_temp_c;
        o[1] = sim.end_soc_pct;
        o[2] = sim.heat_kwh;
        o[3] = sim.charge_time_s;
        o[4] = sim.charge_power_kw;
        o[5] = ph_remaining_km_at_time(&r, t_starts[k]);
        ++ok_count;
    }
    free(snaps);
    return ok_count;
}

/* Route-level helper: total time (s) and total distance (km). */
int ph_route_totals(const double *segs, int n_segs, double *out)
{
    PhSeg s[PH_MAX_SEGS];
    if (segs == NULL || out == NULL || n_segs <= 0 || n_segs > PH_MAX_SEGS) {
        return 0;
    }
    for (int i = 0; i < n_segs; ++i) {
        s[i].s_km       = segs[i * 4 + 0];
        s[i].v_kmh      = segs[i * 4 + 1];
        s[i].p_drive_kw = segs[i * 4 + 2];
        s[i].t_env_c    = segs[i * 4 + 3];
    }
    PhRoute r;
    if (!ph_route_build(&r, s, n_segs)) return 0;
    out[0] = r.total_time_s;
    out[1] = r.total_km;
    return 1;
}
