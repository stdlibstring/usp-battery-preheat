/*
 * preheat_model.h — Battery preheat physics kernel (single source of truth)
 *
 * Extracted VERBATIM from sdk/student_template/student_solution.c so that the
 * offline Oracle generator and the later ML/RL environment (physics.dll) use
 * exactly the same arithmetic as the competition submission:
 *   - 3 MAPs (R_int / U_oc / P_charge) with bilinear interpolation
 *   - Rint low-current root solved in numerically stable form
 *   - coupled thermal / SOC explicit-Euler step
 *   - constant-power charging time from the END state (PDF section 3.4)
 *
 * float literals are kept as float and promoted to double at the exact same
 * places as in student_solution.c, so results are bit-identical.
 *
 * GATE: oracle_generator --selftest must reproduce the official mock result
 *   (342 feasible / 35.32 km / 20.00 C / 15.95 % / 3.1017 kWh / 947.19 s)
 * before any batch generation is allowed.
 */
#ifndef PREHEAT_MODEL_H
#define PREHEAT_MODEL_H

#include <float.h>
#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Physical constants (competition spec section 3.5)                   */
/* float literals on purpose: they promote exactly like the submission */
/* ------------------------------------------------------------------ */
#define PH_M_BAT_KG       400.0f    /* battery pack mass, kg            */
#define PH_CP_J_KG_C     1000.0f    /* specific heat, J/(kg*C)          */
#define PH_C_NOM_AH       100.0f    /* nominal capacity, Ah             */
#define PH_U_NOM_V        350.0f    /* nominal voltage, V               */
#define PH_P_HEAT_MAX_W  6000.0f    /* max heater electrical power, W   */
#define PH_ETA_HEAT         0.95f   /* electro-thermal efficiency       */
#define PH_H0_W_C          50.0f    /* static heat transfer coeff, W/C  */
#define PH_KV_W_C_KMH       0.5f    /* speed-dependent coeff, W/(C*km/h)*/
#define PH_T_LOW_C          20.0f   /* target temp lower bound, C       */
#define PH_T_HIGH_C         25.0f   /* target temp upper bound, C       */
#define PH_SOC_MIN_PCT      10.0f   /* minimum arrival SOC, %           */
#define PH_SOC_TARGET_PCT   80.0f   /* charge target SOC, %             */

#define PH_NUM_EPS          1.0e-9

/* ------------------------------------------------------------------ */
/* MAPs                                                                 */
/* ------------------------------------------------------------------ */
static const float PH_R_INT_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
static const float PH_R_INT_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float PH_R_INT_MAP[7][8] = {
    {0.150f, 0.120f, 0.100f, 0.090f, 0.085f, 0.082f, 0.080f, 0.078f},
    {0.100f, 0.085f, 0.075f, 0.070f, 0.065f, 0.062f, 0.060f, 0.058f},
    {0.070f, 0.060f, 0.052f, 0.048f, 0.045f, 0.043f, 0.042f, 0.041f},
    {0.050f, 0.042f, 0.038f, 0.035f, 0.033f, 0.032f, 0.031f, 0.030f},
    {0.038f, 0.032f, 0.029f, 0.027f, 0.025f, 0.024f, 0.023f, 0.023f},
    {0.032f, 0.028f, 0.025f, 0.023f, 0.022f, 0.021f, 0.020f, 0.020f},
    {0.028f, 0.025f, 0.022f, 0.021f, 0.020f, 0.019f, 0.019f, 0.018f},
};

static const float PH_U_OC_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
static const float PH_U_OC_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float PH_U_OC_MAP[7][8] = {
    {295.0f, 320.0f, 340.0f, 352.0f, 365.0f, 375.0f, 382.0f, 410.0f},
    {298.0f, 323.0f, 343.0f, 355.0f, 368.0f, 378.0f, 385.0f, 413.0f},
    {300.0f, 325.0f, 345.0f, 357.0f, 370.0f, 380.0f, 387.0f, 415.0f},
    {302.0f, 327.0f, 347.0f, 359.0f, 372.0f, 382.0f, 389.0f, 417.0f},
    {303.0f, 328.0f, 348.0f, 360.0f, 373.0f, 383.0f, 390.0f, 418.0f},
    {304.0f, 329.0f, 349.0f, 361.0f, 374.0f, 384.0f, 391.0f, 419.0f},
    {305.0f, 330.0f, 350.0f, 362.0f, 375.0f, 385.0f, 392.0f, 420.0f},
};

static const float PH_P_CHA_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 23.0f, 30.0f, 40.0f};
static const float PH_P_CHA_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float PH_P_CHA_MAP[7][8] = {
    {10.0f, 12.0f, 15.0f, 18.0f, 20.0f, 18.0f, 15.0f, 10.0f},
    {20.0f, 25.0f, 30.0f, 35.0f, 38.0f, 35.0f, 28.0f, 20.0f},
    {40.0f, 50.0f, 60.0f, 65.0f, 68.0f, 65.0f, 55.0f, 40.0f},
    {60.0f, 70.0f, 80.0f, 85.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {75.0f, 85.0f, 90.0f, 92.0f, 90.0f, 88.0f, 75.0f, 55.0f},
    {70.0f, 80.0f, 88.0f, 90.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {50.0f, 60.0f, 70.0f, 72.0f, 70.0f, 65.0f, 55.0f, 40.0f},
};

/* ------------------------------------------------------------------ */
/* Route representation                                                 */
/* ------------------------------------------------------------------ */
#define PH_MAX_SEGS 16

typedef struct {
    double s_km;       /* segment distance, km           */
    double v_kmh;      /* average speed, km/h            */
    double p_drive_kw; /* drive power, kW (discharge +)  */
    double t_env_c;    /* ambient temperature, C         */
} PhSeg;

typedef struct {
    PhSeg segs[PH_MAX_SEGS];
    int    n;
    double total_km;
    double total_time_s;
    double seg_t0[PH_MAX_SEGS + 1]; /* cumulative segment start times, s */
} PhRoute;

static int ph_route_build(PhRoute *r, const PhSeg *segs, int n)
{
    if (r == NULL || segs == NULL || n <= 0 || n > PH_MAX_SEGS) return 0;
    r->n = n;
    r->total_km = 0.0;
    r->total_time_s = 0.0;
    r->seg_t0[0] = 0.0;
    for (int i = 0; i < n; ++i) {
        const double s = segs[i].s_km;
        const double v = segs[i].v_kmh;
        if (!isfinite(s) || !isfinite(v) || s <= 0.0 || v <= 0.0) return 0;
        r->segs[i] = segs[i];
        r->total_km += s;
        r->total_time_s += s / v * 3600.0;
        r->seg_t0[i + 1] = r->total_time_s;
    }
    return (r->total_km > 0.0 && r->total_time_s > 0.0);
}

/* Remaining distance (km from charger) at absolute route time t, s. */
static double ph_remaining_km_at_time(const PhRoute *r, double t_s)
{
    if (t_s <= 0.0) return r->total_km;
    if (t_s >= r->total_time_s) return 0.0;
    for (int i = 0; i < r->n; ++i) {
        if (t_s < r->seg_t0[i + 1] || i == r->n - 1) {
            const double local_s = t_s - r->seg_t0[i];
            const double traveled = r->segs[i].v_kmh * local_s / 3600.0;
            double rem = r->total_km;
            for (int j = 0; j < i; ++j) rem -= r->segs[j].s_km;
            rem -= traveled;
            return (rem > 0.0) ? rem : 0.0;
        }
    }
    return 0.0;
}

/* ------------------------------------------------------------------ */
/* MAP interpolation (verbatim port of student_solution.c)              */
/* ------------------------------------------------------------------ */
static double ph_clamp(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static int ph_lower_index(const float *axis, int count, double value)
{
    if (value <= axis[0]) return 0;
    if (value >= axis[count - 1]) return count - 2;
    for (int i = 0; i < count - 1; ++i) {
        if (value <= axis[i + 1]) return i;
    }
    return count - 2;
}

static double ph_interp_map(const float temp_axis[7],
                            const float soc_axis[8],
                            const float map[7][8],
                            double temp_c,
                            double soc_pct)
{
    const double temp = ph_clamp(temp_c, temp_axis[0], temp_axis[6]);
    const double soc  = ph_clamp(soc_pct, soc_axis[0], soc_axis[7]);
    const int ti = ph_lower_index(temp_axis, 7, temp);
    const int si = ph_lower_index(soc_axis, 8, soc);

    const double tr = (temp - temp_axis[ti]) / (temp_axis[ti + 1] - temp_axis[ti]);
    const double sr = (soc - soc_axis[si]) / (soc_axis[si + 1] - soc_axis[si]);

    const double lower = map[ti][si] + sr * (map[ti][si + 1] - map[ti][si]);
    const double upper = map[ti + 1][si] + sr * (map[ti + 1][si + 1] - map[ti + 1][si]);
    return lower + tr * (upper - lower);
}

static double ph_resistance_ohm(double temp_c, double soc_pct)
{
    return ph_interp_map(PH_R_INT_TEMPS, PH_R_INT_SOCS, PH_R_INT_MAP, temp_c, soc_pct);
}

static double ph_uoc_v(double temp_c, double soc_pct)
{
    return ph_interp_map(PH_U_OC_TEMPS, PH_U_OC_SOCS, PH_U_OC_MAP, temp_c, soc_pct);
}

static double ph_charge_power_kw(double temp_c, double soc_pct)
{
    return ph_interp_map(PH_P_CHA_TEMPS, PH_P_CHA_SOCS, PH_P_CHA_MAP, temp_c, soc_pct);
}

/* Rint low-current root, numerically stable form (discharge power >= 0). */
static int ph_discharge_current(double power_w, double uoc_v, double r_ohm,
                                double *out_a)
{
    if (out_a == NULL || uoc_v <= 0.0 || r_ohm < 0.0) return 0;
    if (r_ohm < PH_NUM_EPS) {
        *out_a = power_w / uoc_v;
        return isfinite(*out_a);
    }
    double disc = uoc_v * uoc_v - 4.0 * r_ohm * power_w;
    if (disc < 0.0) {
        if (disc > -1.0e-6) disc = 0.0;
        else return 0;
    }
    const double denom = uoc_v + sqrt(disc);
    if (fabs(denom) < PH_NUM_EPS) return 0;
    *out_a = 2.0 * power_w / denom;
    return isfinite(*out_a);
}

/* ------------------------------------------------------------------ */
/* One explicit-Euler integration step (verbatim semantics)             */
/* heating: 0 = PTC off, 1 = PTC on. Returns 0 on numerical failure.    */
/* ------------------------------------------------------------------ */
static int ph_step(const PhSeg *seg, int heating, double dt_s,
                   double *temp_c, double *soc_pct, double *heat_kwh)
{
    const double heating_power_w = heating ? (double)PH_P_HEAT_MAX_W : 0.0;
    const double r_ohm = ph_resistance_ohm(*temp_c, *soc_pct);
    const double uoc_v = ph_uoc_v(*temp_c, *soc_pct);
    const double battery_power_w = seg->p_drive_kw * 1000.0 + heating_power_w;

    double current_a = 0.0;
    if (!ph_discharge_current(battery_power_w, uoc_v, r_ohm, &current_a)) {
        return 0;
    }

    const double resist_heat_w = current_a * current_a * r_ohm;
    const double ptc_heat_w = PH_ETA_HEAT * heating_power_w;
    const double h_w_per_c = PH_H0_W_C + PH_KV_W_C_KMH * seg->v_kmh;
    const double ambient_loss_w = h_w_per_c * (*temp_c - seg->t_env_c);

    *temp_c += (resist_heat_w + ptc_heat_w - ambient_loss_w) * dt_s
               / (PH_M_BAT_KG * PH_CP_J_KG_C);
    *soc_pct -= current_a * dt_s / (PH_C_NOM_AH * 3600.0) * 100.0;
    *heat_kwh += heating_power_w * dt_s / 3.6e6;

    if (!isfinite(*temp_c) || !isfinite(*soc_pct) || *soc_pct < -1.0) {
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Prefix cache: the heater-free trajectory [0, t_start) is IDENTICAL   */
/* for every candidate start time, so simulate it once per case and     */
/* snapshot the state at every integration boundary. A candidate then   */
/* costs one partial step + heated suffix only — exactly reproducing    */
/* student_solution.c's "split the step at switch-on" arithmetic.       */
/* ------------------------------------------------------------------ */
typedef struct {
    double t_s;        /* absolute time at step start            */
    double temp_c;
    double soc_pct;
    int    seg;        /* segment index the step belongs to      */
    double seg_rem_km; /* remaining km inside that segment       */
} PhSnap;

typedef struct {
    PhSnap *snaps;
    int     n;
    int     cap;
    /* final no-heating state at route end (for start >= total_time) */
    double  end_temp_c;
    double  end_soc_pct;
} PhPrefix;

/* Simulate the whole route with heater OFF at step dt; snapshot every
 * integration boundary (per-segment grids + partial steps at segment
 * ends — candidate independent). Returns 0 on numerical failure. */
static int ph_prefix_build(const PhRoute *r, double t0_c, double soc0_pct,
                           double dt, PhSnap *snaps, int snap_cap, int *n_out,
                           double *end_temp, double *end_soc)
{
    double temp_c = t0_c;
    double soc_pct = soc0_pct;
    double heat_kwh = 0.0; /* stays 0: heater off */
    int count = 0;

    for (int i = 0; i < r->n; ++i) {
        double seg_rem_km = r->segs[i].s_km;
        double elapsed_s = r->seg_t0[i];

        while (seg_rem_km > 1.0e-10) {
            if (count >= snap_cap) return 0;
            snaps[count].t_s = elapsed_s;
            snaps[count].temp_c = temp_c;
            snaps[count].soc_pct = soc_pct;
            snaps[count].seg = i;
            snaps[count].seg_rem_km = seg_rem_km;
            ++count;

            double step_s = fmin(dt, seg_rem_km / r->segs[i].v_kmh * 3600.0);
            if (step_s <= PH_NUM_EPS) return 0;
            if (!ph_step(&r->segs[i], 0, step_s, &temp_c, &soc_pct, &heat_kwh)) {
                return 0;
            }

            const double traveled_km = r->segs[i].v_kmh * step_s / 3600.0;
            seg_rem_km = fmax(0.0, seg_rem_km - traveled_km);
            elapsed_s += step_s;
        }
    }

    *n_out = count;
    *end_temp = temp_c;
    *end_soc = soc_pct;
    return 1;
}

typedef struct {
    int    valid;
    double end_temp_c;
    double end_soc_pct;
    double heat_kwh;
    double charge_power_kw;
    double charge_time_s;
} PhSimResult;

/* Simulate ONE candidate: heated suffix from t_start to route end.
 * Semantics are identical to simulate_candidate() in student_solution.c:
 * one partial heater-off step to reach t_start, then heater on with
 * per-segment grids. t_start >= total_time_s means "never heat". */
static PhSimResult ph_simulate_candidate(const PhRoute *r,
                                         const PhPrefix *prefix,
                                         double dt,
                                         double t_start)
{
    PhSimResult res;
    res.valid = 0;
    res.end_temp_c = 0.0;
    res.end_soc_pct = 0.0;
    res.heat_kwh = 0.0;
    res.charge_power_kw = 0.0;
    res.charge_time_s = DBL_MAX;

    if (t_start < 0.0) t_start = 0.0;
    if (t_start > r->total_time_s) t_start = r->total_time_s;

    if (t_start >= r->total_time_s - PH_NUM_EPS) {
        /* no heating at all: the no-heating prefix already is the answer */
        res.end_temp_c = prefix->end_temp_c;
        res.end_soc_pct = prefix->end_soc_pct;
        res.heat_kwh = 0.0;
        res.valid = 1;
        res.charge_power_kw = ph_charge_power_kw(res.end_temp_c, res.end_soc_pct);
        const double e_kwh = fmax(0.0, (PH_SOC_TARGET_PCT - res.end_soc_pct) / 100.0 *
                                 PH_C_NOM_AH * PH_U_NOM_V / 1000.0);
        res.charge_time_s = res.charge_power_kw > PH_NUM_EPS
                                ? e_kwh / res.charge_power_kw * 3600.0
                                : DBL_MAX;
        return res;
    }

    /* locate last snapshot at or before t_start */
    int lo = 0, hi = prefix->n - 1, idx = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (prefix->snaps[mid].t_s <= t_start + PH_NUM_EPS) {
            idx = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (idx < 0) return res;

    const PhSnap *snap = &prefix->snaps[idx];
    double temp_c = snap->temp_c;
    double soc_pct = snap->soc_pct;
    double heat_kwh = 0.0;
    int seg = snap->seg;
    double seg_rem_km = snap->seg_rem_km;
    double elapsed_s = snap->t_s;

    /* partial heater-off step to land exactly on t_start */
    const double dt0 = t_start - snap->t_s;
    if (dt0 > PH_NUM_EPS) {
        if (!ph_step(&r->segs[seg], 0, dt0, &temp_c, &soc_pct, &heat_kwh)) {
            return res;
        }
        const double traveled = r->segs[seg].v_kmh * dt0 / 3600.0;
        seg_rem_km = fmax(0.0, seg_rem_km - traveled);
        elapsed_s += dt0;
    }

    /* heated suffix through the rest of the route */
    while (seg < r->n) {
        while (seg_rem_km > 1.0e-10) {
            double step_s = fmin(dt, seg_rem_km / r->segs[seg].v_kmh * 3600.0);
            if (step_s <= PH_NUM_EPS) return res;
            if (!ph_step(&r->segs[seg], 1, step_s, &temp_c, &soc_pct, &heat_kwh)) {
                return res;
            }
            const double traveled = r->segs[seg].v_kmh * step_s / 3600.0;
            seg_rem_km = fmax(0.0, seg_rem_km - traveled);
            elapsed_s += step_s;
        }
        ++seg;
        if (seg < r->n) seg_rem_km = r->segs[seg].s_km;
    }

    res.valid = 1;
    res.end_temp_c = temp_c;
    res.end_soc_pct = soc_pct;
    res.heat_kwh = heat_kwh;
    res.charge_power_kw = ph_charge_power_kw(temp_c, soc_pct);
    const double e_kwh = fmax(0.0, (PH_SOC_TARGET_PCT - soc_pct) / 100.0 *
                              PH_C_NOM_AH * PH_U_NOM_V / 1000.0);
    res.charge_time_s = res.charge_power_kw > PH_NUM_EPS
                            ? e_kwh / res.charge_power_kw * 3600.0
                            : DBL_MAX;
    return res;
}

/* Constraint violation penalty (same shape as student_solution.c). */
static inline double ph_constraint_penalty(double t_end, double soc_end,
                                           double t_low, double t_high,
                                           double soc_min)
{
    double tv = 0.0, sv = 0.0;
    if (t_end < t_low) tv = t_low - t_end;
    else if (t_end > t_high) tv = t_end - t_high;
    if (soc_end < soc_min) sv = soc_min - soc_end;
    return tv * tv + 10.0 * sv * sv;
}

#ifdef __cplusplus
}
#endif

#endif /* PREHEAT_MODEL_H */
