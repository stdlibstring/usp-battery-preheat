/**
 * @file student_solution.c
 * @brief BEV Battery Preheat Decision Algorithm �?Student Template
 *
 * Outputs:
 *   start_distance  �?optimal preheat start distance (km from charger)
 *   T_end_opt       �?predicted battery temp at charger arrival (C)
 *   SOC_end_opt     �?predicted SOC at charger arrival (%)
 *   E_heat_opt      �?total preheat energy consumption (kWh)
 *   chrg_time_s   — predicted charging time to 80% SOC (seconds)
 *
 * How to use:
 *   1. Read initial battery state and navigation data via the provided interfaces.
 *   2. Implement your algorithm between the "TODO" markers below.
 *   3. Use BattChrgPreHeatg_ntf*() to notify the system of your results.
 *   4. Compile and run:
 *        mingw32-make
 *      or
 *        gcc -Wall -O2 -std=c11 -o run.exe student_solution.c \
 *            -I./include -L./lib -lcompetition_mock -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "usp_api.h"
#include "mlp_weights.h"   /* MLP 权重 (ml/export_mlp_c.py 生成) */

/* ========================================================================
 *  Physical constants & parameters (from competition spec Section 3.5)
 *
 *  These are REFERENCE DATA for your algorithm. Do not modify.
 * ======================================================================== */

/* Thermal model */
#define M_BAT       400.0f    /* kg �?battery pack mass */
#define CP          1000.0f   /* J/(kg·C) �?specific heat capacity */

/* Electrical model */
#define C_NOM_AH    100.0f    /* Ah �?nominal capacity */
#define U_NOM       350.0f    /* V �?nominal voltage */

/* PTC heater */
#define P_HEAT_MAX  6000.0f   /* W �?max heater power */
#define ETA_HEAT    0.95f     /* electro-thermal efficiency */

/* Heat transfer to ambient */
#define H0          50.0f     /* W/C �?static heat transfer coeff */
#define KV          0.5f      /* W/(C·km/h) �?speed-dependent coeff */

/* Target constraints */
#define T_OPT_LOW   20.0f     /* C �?target temp lower bound */
#define T_OPT_HIGH  25.0f     /* C �?target temp upper bound */
#define SOC_TARGET  0.80f     /* fast-charge target SOC (for reference) */
#define SOC_MIN     0.10f     /* minimum safe SOC at arrival */


/* ========================================================================
 *  R_int(temperature x SOC) MAP [Ohm]
 *
 *  Rows: -20, -10, 0, 10, 20, 30, 40 C
 *  Cols: 0%, 10%, 20%, 30%, 50%, 70%, 90%, 100%
 *
 *  Use bilinear interpolation to get R_int at any (T, SOC).
 * ======================================================================== */
static const float R_INT_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
static const float R_INT_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float R_INT_MAP[7][8] = {
    {0.150f, 0.120f, 0.100f, 0.090f, 0.085f, 0.082f, 0.080f, 0.078f},
    {0.100f, 0.085f, 0.075f, 0.070f, 0.065f, 0.062f, 0.060f, 0.058f},
    {0.070f, 0.060f, 0.052f, 0.048f, 0.045f, 0.043f, 0.042f, 0.041f},
    {0.050f, 0.042f, 0.038f, 0.035f, 0.033f, 0.032f, 0.031f, 0.030f},
    {0.038f, 0.032f, 0.029f, 0.027f, 0.025f, 0.024f, 0.023f, 0.023f},
    {0.032f, 0.028f, 0.025f, 0.023f, 0.022f, 0.021f, 0.020f, 0.020f},
    {0.028f, 0.025f, 0.022f, 0.021f, 0.020f, 0.019f, 0.019f, 0.018f},
};

/* ========================================================================
 *  U_oc(temperature x SOC) MAP [V, pack voltage]
 *
 *  You can either call EMS_HVBatt_getVolt() for current operating point,
 *  or use bilinear interpolation on this MAP for algorithm calculations
 *  (e.g., I = P/V). You need to implement the interpolation yourself.
 * ======================================================================== */

static const float U_OC_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
static const float U_OC_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float UOC_MAP[7][8] = {
    {295.0f, 320.0f, 340.0f, 352.0f, 365.0f, 375.0f, 382.0f, 410.0f},
    {298.0f, 323.0f, 343.0f, 355.0f, 368.0f, 378.0f, 385.0f, 413.0f},
    {300.0f, 325.0f, 345.0f, 357.0f, 370.0f, 380.0f, 387.0f, 415.0f},
    {302.0f, 327.0f, 347.0f, 359.0f, 372.0f, 382.0f, 389.0f, 417.0f},
    {303.0f, 328.0f, 348.0f, 360.0f, 373.0f, 383.0f, 390.0f, 418.0f},
    {304.0f, 329.0f, 349.0f, 361.0f, 374.0f, 384.0f, 391.0f, 419.0f},
    {305.0f, 330.0f, 350.0f, 362.0f, 375.0f, 385.0f, 392.0f, 420.0f},
};


/* ========================================================================
 *  P_charge(temperature x SOC) MAP [kW, max charging power]
 *
 *  Rows: -20, -10, 0, 10, 23, 30, 40 C
 *  Cols: 0%, 10%, 20%, 30%, 50%, 70%, 90%, 100%
 * ======================================================================== */
static const float P_CHARGE_TEMPS[7] = {-20.0f, -10.0f, 0.0f, 10.0f, 23.0f, 30.0f, 40.0f};
static const float P_CHARGE_SOCS[8]  = {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};
static const float P_CHARGE_MAP[7][8] = {
    {10.0f, 12.0f, 15.0f, 18.0f, 20.0f, 18.0f, 15.0f, 10.0f},
    {20.0f, 25.0f, 30.0f, 35.0f, 38.0f, 35.0f, 28.0f, 20.0f},
    {40.0f, 50.0f, 60.0f, 65.0f, 68.0f, 65.0f, 55.0f, 40.0f},
    {60.0f, 70.0f, 80.0f, 85.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {75.0f, 85.0f, 90.0f, 92.0f, 90.0f, 88.0f, 75.0f, 55.0f},
    {70.0f, 80.0f, 88.0f, 90.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {50.0f, 60.0f, 70.0f, 72.0f, 70.0f, 65.0f, 55.0f, 40.0f},
};

/* ========================================================================
 *  Navigation segment data
 * ======================================================================== */
#define MAX_SEGS  16

typedef struct {
    float s_km;       /* segment distance, km */
    float v_kmh;      /* average speed, km/h */
    float P_drive_kW; /* drive power, kW (positive = discharge) */
    float T_env_C;    /* ambient temperature, C */
} NavSeg;

static NavSeg   g_segs[MAX_SEGS];
static int      g_n_segs = 0;

/* Load navigation data from VehPwrPred_getPwrPred() interface */
static void load_nav_segments(void)
{
    PwrPredList_stru pred;
    VehPwrPred_getPwrPred(&pred);
    g_n_segs = (pred.num < MAX_SEGS) ? pred.num : MAX_SEGS;

    for (int i = 0; i < g_n_segs; i++) {
        g_segs[i].s_km       = pred.pwrPredList[i].length;
        g_segs[i].v_kmh      = pred.pwrPredList[i].avgSpd;
        g_segs[i].P_drive_kW = pred.pwrPredList[i].drvPwr;
        g_segs[i].T_env_C    = pred.pwrPredList[i].ambTemp;
    }
}

/* ========================================================================
 *  TODO: Implement your algorithm functions below
 *
 * ======================================================================== */
/* --- Your algorithm code starts here --- */

#define SIM_DT_S             0.5
#define CANDIDATE_STEP_S     1.0
#define NUM_EPS              1.0e-9
static long g_sim_count = 0;   /* 物理仿真调用计数 (性能统计) */

typedef struct {
    double total_distance_km;
    double total_time_s;
} RouteSummary;

typedef struct {
    int valid;
    int feasible;
    double start_time_s;
    double start_distance_km;
    double end_temp_c;
    double end_soc_pct;
    double heat_energy_kwh;
    double charge_power_kw;
    double charge_time_s;
} SimResult;

typedef struct {
    SimResult sim;
    double score;
} Candidate;

static double clamp_value(double value, double low, double high)
{
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static int lower_axis_index(const float *axis, int count, double value)
{
    if (value <= axis[0]) return 0;
    if (value >= axis[count - 1]) return count - 2;

    for (int i = 0; i < count - 1; ++i) {
        if (value <= axis[i + 1]) return i;
    }
    return count - 2;
}

static double interpolate_map(const float temp_axis[7],
                              const float soc_axis[8],
                              const float map[7][8],
                              double temp_c,
                              double soc_pct)
{
    const double temp = clamp_value(temp_c, temp_axis[0], temp_axis[6]);
    const double soc = clamp_value(soc_pct, soc_axis[0], soc_axis[7]);
    const int ti = lower_axis_index(temp_axis, 7, temp);
    const int si = lower_axis_index(soc_axis, 8, soc);

    const double temp_ratio = (temp - temp_axis[ti]) /
                              (temp_axis[ti + 1] - temp_axis[ti]);
    const double soc_ratio = (soc - soc_axis[si]) /
                             (soc_axis[si + 1] - soc_axis[si]);

    const double lower = map[ti][si] + soc_ratio *
                         (map[ti][si + 1] - map[ti][si]);
    const double upper = map[ti + 1][si] + soc_ratio *
                         (map[ti + 1][si + 1] - map[ti + 1][si]);
    return lower + temp_ratio * (upper - lower);
}

static double resistance_ohm(double temp_c, double soc_pct)
{
    return interpolate_map(R_INT_TEMPS, R_INT_SOCS, R_INT_MAP,
                           temp_c, soc_pct);
}

static double open_circuit_voltage_v(double temp_c, double soc_pct)
{
    return interpolate_map(U_OC_TEMPS, U_OC_SOCS, UOC_MAP,
                           temp_c, soc_pct);
}

static double charge_power_kw(double temp_c, double soc_pct)
{
    return interpolate_map(P_CHARGE_TEMPS, P_CHARGE_SOCS, P_CHARGE_MAP,
                           temp_c, soc_pct);
}

static int calculate_discharge_current(double power_w,
                                       double uoc_v,
                                       double resistance,
                                       double *current_a)
{
    if (current_a == NULL || uoc_v <= 0.0 || resistance < 0.0) return 0;

    if (resistance < NUM_EPS) {
        *current_a = power_w / uoc_v;
        return isfinite(*current_a);
    }

    double discriminant = uoc_v * uoc_v - 4.0 * resistance * power_w;
    if (discriminant < 0.0) {
        if (discriminant > -1.0e-6) discriminant = 0.0;
        else return 0;
    }

    /* Low-current root, written in a numerically stable form. */
    const double denominator = uoc_v + sqrt(discriminant);
    if (fabs(denominator) < NUM_EPS) return 0;

    *current_a = 2.0 * power_w / denominator;
    return isfinite(*current_a);
}

static int summarize_route(RouteSummary *summary)
{
    if (summary == NULL || g_n_segs <= 0) return 0;

    summary->total_distance_km = 0.0;
    summary->total_time_s = 0.0;

    for (int i = 0; i < g_n_segs; ++i) {
        const double distance = g_segs[i].s_km;
        const double speed = g_segs[i].v_kmh;
        if (!isfinite(distance) || !isfinite(speed) ||
            distance < 0.0 || speed <= 0.0) {
            return 0;
        }
        summary->total_distance_km += distance;
        summary->total_time_s += distance / speed * 3600.0;
    }

    return summary->total_distance_km > 0.0 && summary->total_time_s > 0.0;
}

static double remaining_distance_at_time(const RouteSummary *route,
                                         double query_time_s)
{
    double remaining_km = route->total_distance_km;
    double elapsed_s = 0.0;

    for (int i = 0; i < g_n_segs; ++i) {
        const double segment_time_s =
            (double)g_segs[i].s_km / (double)g_segs[i].v_kmh * 3600.0;

        if (query_time_s >= elapsed_s + segment_time_s) {
            remaining_km -= g_segs[i].s_km;
            elapsed_s += segment_time_s;
        } else {
            const double local_time_s = clamp_value(query_time_s - elapsed_s,
                                                    0.0, segment_time_s);
            remaining_km -= (double)g_segs[i].v_kmh * local_time_s / 3600.0;
            break;
        }
    }

    return fmax(0.0, remaining_km);
}

static SimResult simulate_candidate(const RouteSummary *route,
                                    double initial_temp_c,
                                    double initial_soc_pct,
                                    double requested_start_time_s)
{
    ++g_sim_count;
    SimResult result = {0};
    result.start_time_s = clamp_value(requested_start_time_s,
                                     0.0, route->total_time_s);
    result.start_distance_km =
        remaining_distance_at_time(route, result.start_time_s);

    double temp_c = initial_temp_c;
    double soc_pct = initial_soc_pct;
    double heat_energy_kwh = 0.0;
    double elapsed_s = 0.0;

    for (int seg_index = 0; seg_index < g_n_segs; ++seg_index) {
        const NavSeg *segment = &g_segs[seg_index];
        double segment_remaining_km = segment->s_km;

        while (segment_remaining_km > 1.0e-10) {
            const int heating_on =
                elapsed_s + NUM_EPS >= result.start_time_s &&
                result.start_time_s < route->total_time_s - NUM_EPS;

            double dt_s = fmin(SIM_DT_S,
                               segment_remaining_km / segment->v_kmh * 3600.0);

            /* Do not integrate across the exact heater switch-on instant. */
            if (!heating_on &&
                result.start_time_s > elapsed_s + NUM_EPS &&
                result.start_time_s < elapsed_s + dt_s - NUM_EPS) {
                dt_s = result.start_time_s - elapsed_s;
            }

            if (dt_s <= NUM_EPS) return result;

            const double heating_power_w = heating_on ? P_HEAT_MAX : 0.0;
            const double resistance = resistance_ohm(temp_c, soc_pct);
            const double uoc_v = open_circuit_voltage_v(temp_c, soc_pct);
            const double battery_power_w =
                (double)segment->P_drive_kW * 1000.0 + heating_power_w;

            double current_a = 0.0;
            if (!calculate_discharge_current(battery_power_w, uoc_v,
                                             resistance, &current_a)) {
                return result;
            }

            const double resistance_heat_w = current_a * current_a * resistance;
            const double ptc_heat_w = ETA_HEAT * heating_power_w;
            const double heat_transfer_w_per_c =
                H0 + KV * (double)segment->v_kmh;
            const double ambient_loss_w = heat_transfer_w_per_c *
                                          (temp_c - segment->T_env_C);

            temp_c += (resistance_heat_w + ptc_heat_w - ambient_loss_w) *
                      dt_s / (M_BAT * CP);
            soc_pct -= current_a * dt_s /
                       (C_NOM_AH * 3600.0) * 100.0;
            heat_energy_kwh += heating_power_w * dt_s / 3.6e6;

            const double traveled_km =
                (double)segment->v_kmh * dt_s / 3600.0;
            segment_remaining_km = fmax(0.0,
                                        segment_remaining_km - traveled_km);
            elapsed_s += dt_s;

            if (!isfinite(temp_c) || !isfinite(soc_pct) || soc_pct < -1.0) {
                return result;
            }
        }
    }

    result.valid = 1;
    result.end_temp_c = temp_c;
    result.end_soc_pct = soc_pct;
    result.heat_energy_kwh = heat_energy_kwh;
    result.charge_power_kw = charge_power_kw(temp_c, soc_pct);

    const double target_soc_pct = SOC_TARGET * 100.0;
    const double energy_to_target_kwh =
        fmax(0.0, (target_soc_pct - soc_pct) / 100.0 *
                  C_NOM_AH * U_NOM / 1000.0);
    result.charge_time_s = result.charge_power_kw > NUM_EPS
                               ? energy_to_target_kwh /
                                 result.charge_power_kw * 3600.0
                               : DBL_MAX;

    result.feasible =
        temp_c >= T_OPT_LOW && temp_c <= T_OPT_HIGH &&
        soc_pct >= SOC_MIN * 100.0;
    return result;
}

static double constraint_penalty(const SimResult *sim)
{
    double temp_violation = 0.0;
    double soc_violation = 0.0;

    if (sim->end_temp_c < T_OPT_LOW) {
        temp_violation = T_OPT_LOW - sim->end_temp_c;
    } else if (sim->end_temp_c > T_OPT_HIGH) {
        temp_violation = sim->end_temp_c - T_OPT_HIGH;
    }

    if (sim->end_soc_pct < SOC_MIN * 100.0) {
        soc_violation = SOC_MIN * 100.0 - sim->end_soc_pct;
    }

    return temp_violation * temp_violation +
           10.0 * soc_violation * soc_violation;
}

/*
 * Return values:
 *   1: a feasible optimum was found
 *   0: no feasible strategy; best constraint-violation fallback returned
 *  -1: route or simulation invalid
 */
static int optimize_preheat(double initial_temp_c,
                            double initial_soc_pct,
                            Candidate *best,
                            int *feasible_count)
{
    RouteSummary route;
    if (best == NULL || feasible_count == NULL || !summarize_route(&route)) {
        return -1;
    }

    const int max_candidates =
        (int)ceil(route.total_time_s / CANDIDATE_STEP_S) + 2;
    Candidate *feasible =
        (Candidate *)calloc((size_t)max_candidates, sizeof(Candidate));
    if (feasible == NULL) return -1;

    int count = 0;
    int have_fallback = 0;
    SimResult fallback = {0};
    double fallback_penalty = DBL_MAX;

    for (int candidate_index = 0; candidate_index < max_candidates;
         ++candidate_index) {
        double start_time_s = candidate_index * CANDIDATE_STEP_S;
        if (start_time_s > route.total_time_s) {
            start_time_s = route.total_time_s;
        }

        SimResult sim = simulate_candidate(&route, initial_temp_c,
                                           initial_soc_pct, start_time_s);
        if (sim.valid) {
            const double penalty = constraint_penalty(&sim);
            if (!have_fallback || penalty < fallback_penalty - NUM_EPS ||
                (fabs(penalty - fallback_penalty) <= NUM_EPS &&
                 sim.heat_energy_kwh < fallback.heat_energy_kwh)) {
                fallback = sim;
                fallback_penalty = penalty;
                have_fallback = 1;
            }

            if (sim.feasible) {
                feasible[count].sim = sim;
                feasible[count].score = 0.0;
                ++count;
            }
        }

        if (start_time_s >= route.total_time_s - NUM_EPS) break;
    }

    *feasible_count = count;
    if (count == 0) {
        free(feasible);
        if (!have_fallback) return -1;
        best->sim = fallback;
        best->score = 0.0;
        return 0;
    }

    double min_charge_time_s = DBL_MAX;
    double max_charge_time_s = -DBL_MAX;
    double min_heat_energy_kwh = DBL_MAX;
    double max_heat_energy_kwh = -DBL_MAX;

    for (int i = 0; i < count; ++i) {
        min_charge_time_s = fmin(min_charge_time_s,
                                 feasible[i].sim.charge_time_s);
        max_charge_time_s = fmax(max_charge_time_s,
                                 feasible[i].sim.charge_time_s);
        min_heat_energy_kwh = fmin(min_heat_energy_kwh,
                                   feasible[i].sim.heat_energy_kwh);
        max_heat_energy_kwh = fmax(max_heat_energy_kwh,
                                   feasible[i].sim.heat_energy_kwh);
    }

    int best_index = 0;
    double best_score = -DBL_MAX;
    for (int i = 0; i < count; ++i) {
        const double charge_range = max_charge_time_s - min_charge_time_s;
        const double energy_range = max_heat_energy_kwh - min_heat_energy_kwh;

        const double charge_score = charge_range > NUM_EPS
            ? 30.0 * (max_charge_time_s - feasible[i].sim.charge_time_s) /
              charge_range
            : 30.0;
        const double energy_score = energy_range > NUM_EPS
            ? 20.0 * (max_heat_energy_kwh - feasible[i].sim.heat_energy_kwh) /
              energy_range
            : 20.0;

        feasible[i].score = charge_score + energy_score;
        if (feasible[i].score > best_score + NUM_EPS ||
            (fabs(feasible[i].score - best_score) <= NUM_EPS &&
             feasible[i].sim.heat_energy_kwh <
                 feasible[best_index].sim.heat_energy_kwh - NUM_EPS) ||
            (fabs(feasible[i].score - best_score) <= NUM_EPS &&
             fabs(feasible[i].sim.heat_energy_kwh -
                  feasible[best_index].sim.heat_energy_kwh) <= NUM_EPS &&
             feasible[i].sim.start_time_s >
                 feasible[best_index].sim.start_time_s)) {
            best_index = i;
            best_score = feasible[i].score;
        }
    }

    *best = feasible[best_index];
    free(feasible);
    return 1;
}


/* ====================================================================== */
/*  MLP-guided search: 神经网络先验缩小区间, 物理仿真在窗口内精搜           */
/*                                                                      */
/*  MLP 前向仅 ~30k 乘加 (微秒级), 把搜索区间从 [0, total_time] 缩到      */
/*  [t_pred ± 60 s]; 窗口内 1 s 网格物理仿真 + 批内得分最大化选点,        */
/*  精度由物理仿真保证; 窗口内无可行时才退回全路线搜索。                  */
/* ====================================================================== */
#define MLP_GUIDE_WINDOW_S   60.0
#define MLP_POOL_MAX         160

static double mlp_predict_start_time(const RouteSummary *route,
                                     double initial_temp_c,
                                     double initial_soc_pct)
{
    if (MLP_IN_DIM != 6 * 4 + 6) return -1.0;   /* 结构不匹配则禁用 */

    /* 30 维特征, 与 ml/train_mlp.py build_features 严格同序:
       24 段展开 + T_init, SOC_init, total_time_s, total_km, mean_T, mean_v */
    double feat[MLP_IN_DIM];
    int k = 0;
    for (int i = 0; i < 6; ++i) {
        if (i < g_n_segs) {
            feat[k++] = g_segs[i].s_km;
            feat[k++] = g_segs[i].v_kmh;
            feat[k++] = g_segs[i].P_drive_kW;
            feat[k++] = g_segs[i].T_env_C;
        } else {
            feat[k++] = 0.0; feat[k++] = 0.0;
            feat[k++] = 0.0; feat[k++] = 0.0;
        }
    }
    double mean_T = 0.0, mean_v = 0.0;
    for (int i = 0; i < g_n_segs; ++i) {
        mean_T += g_segs[i].T_env_C;
        mean_v += g_segs[i].v_kmh;
    }
    if (g_n_segs > 0) { mean_T /= g_n_segs; mean_v /= g_n_segs; }
    feat[k++] = initial_temp_c;
    feat[k++] = initial_soc_pct;
    feat[k++] = route->total_time_s;
    feat[k++] = route->total_distance_km;
    feat[k++] = mean_T;
    feat[k++] = mean_v;

    /* 归一化 + 前向: backbone 3 层 Linear+ReLU, head_t + sigmoid */
    float buf[2][MLP_MAX_WIDTH];
    int cur = 0;
    for (int i = 0; i < MLP_IN_DIM; ++i)
        buf[0][i] = (float)((feat[i] - (double)mlp_mu[i]) / (double)mlp_sd[i]);
    for (int l = 0; l < MLP_N_LAYERS; ++l) {
        const int n_in = MLP_LAYER_IN[l];
        const int n_out = MLP_LAYER_OUT[l];
        const float *src = buf[cur];
        float *dst = buf[1 - cur];
        for (int o = 0; o < n_out; ++o) {
            const float *w = MLP_W[l] + (size_t)o * n_in;
            float acc = MLP_B[l][o];
            for (int i = 0; i < n_in; ++i) acc += w[i] * src[i];
            dst[o] = (l < MLP_N_LAYERS - 1 && acc < 0.0f) ? 0.0f : acc;
        }
        cur = 1 - cur;
    }
    const float frac = 1.0f / (1.0f + expf(-buf[cur][0]));
    return (double)frac * route->total_time_s;
}

/* 返回 1: 窗口内找到可行最优; 0: 窗口内无可行(调用方走全路线兜底); -1: 不可用 */
static int mlp_guided_search(double initial_temp_c,
                             double initial_soc_pct,
                             Candidate *best_out)
{
    RouteSummary route;
    if (!summarize_route(&route)) return -1;
    const double t_pred = mlp_predict_start_time(&route, initial_temp_c,
                                                  initial_soc_pct);
    if (t_pred < 0.0) return 0;

    double lo = t_pred - MLP_GUIDE_WINDOW_S;
    double hi = t_pred + MLP_GUIDE_WINDOW_S;
    if (lo < 0.0) lo = 0.0;
    if (hi > route.total_time_s) hi = route.total_time_s;

    Candidate pool[MLP_POOL_MAX];
    int n = 0;
    for (double t = lo; t <= hi + NUM_EPS; t += CANDIDATE_STEP_S) {
        if (t > hi + NUM_EPS || n >= MLP_POOL_MAX) break;
        SimResult sim = simulate_candidate(&route, initial_temp_c,
                                           initial_soc_pct, t);
        if (sim.valid && sim.feasible) {
            pool[n].sim = sim;
            pool[n].score = 0.0;
            ++n;
        }
    }
    if (n == 0) return 0;

    /* 窗口内批间 min-max 归一化得分选点 (与评测公式同构) */
    double min_tch = DBL_MAX, max_tch = -DBL_MAX;
    double min_e = DBL_MAX, max_e = -DBL_MAX;
    for (int i = 0; i < n; ++i) {
        min_tch = fmin(min_tch, pool[i].sim.charge_time_s);
        max_tch = fmax(max_tch, pool[i].sim.charge_time_s);
        min_e = fmin(min_e, pool[i].sim.heat_energy_kwh);
        max_e = fmax(max_e, pool[i].sim.heat_energy_kwh);
    }
    const double tch_range = max_tch - min_tch;
    const double e_range = max_e - min_e;

    int best = 0;
    double best_score = -DBL_MAX;
    for (int i = 0; i < n; ++i) {
        const double s_t = tch_range > NUM_EPS
            ? 30.0 * (max_tch - pool[i].sim.charge_time_s) / tch_range : 30.0;
        const double s_e = e_range > NUM_EPS
            ? 20.0 * (max_e - pool[i].sim.heat_energy_kwh) / e_range : 20.0;
        pool[i].score = s_t + s_e;
        if (pool[i].score > best_score + NUM_EPS ||
            (fabs(pool[i].score - best_score) <= NUM_EPS &&
             (pool[i].sim.heat_energy_kwh <
                  pool[best].sim.heat_energy_kwh - NUM_EPS ||
              (fabs(pool[i].sim.heat_energy_kwh -
                    pool[best].sim.heat_energy_kwh) <= NUM_EPS &&
               pool[i].sim.start_time_s > pool[best].sim.start_time_s)))) {
            best = i;
            best_score = pool[i].score;
        }
    }
    *best_out = pool[best];
    return 1;
}


/* --- Your algorithm code ends here --- */


/* ========================================================================
 *  Main entry point
 * ======================================================================== */
int main(void)
{
    /* --- Step 1: Read initial battery state --- */
    float T_init = 0.0f;
    EMS_HVBatt_getTempAvg(&T_init);

    float SOC_init = 0.0f;
    EMS_HVBatt_getTargetSOC(&SOC_init);
    SOC_init /= 100.0f;   /* convert % to 0~1 */

    float I_batt = 0.0f;
    EMS_HVBatt_getCurrent(&I_batt);

    float V_batt = 0.0f;
    EMS_HVBatt_getVolt(&V_batt);

    /* Load navigation data */
    load_nav_segments();

    /* Print initial state */
    printf("=== Initial State ===\n");
    printf("  Battery temp:     %.1f C\n", T_init);
    printf("  Battery SOC:      %.1f %%\n", SOC_init * 100.0f);
    printf("  Battery current:  %.1f A\n", I_batt);
    printf("  Battery voltage:  %.1f V\n", V_batt);
    printf("  Route segments:   %d\n", g_n_segs);

    printf("\n=== Route Segments ===\n");
    printf("  %-6s %-10s %-10s %-12s %-12s\n",
           "Seg", "Dist(km)", "Spd(km/h)", "Pdrive(kW)", "Tamb(C)");
    for (int i = 0; i < g_n_segs; i++) {
        printf("  %-6d %-10.3f %-10.3f %-12.3f %-12.3f\n",
               i + 1,
               g_segs[i].s_km,
               g_segs[i].v_kmh,
               g_segs[i].P_drive_kW,
               g_segs[i].T_env_C);
    }

    /* --- Step 2: Compute optimal preheat start distance --- */
    printf("\n=== Algorithm (implement your solution above) ===\n");

    /* TODO: Replace the placeholder values below with your algorithm's output.
     *
     * Required outputs:
     *   start_distance  — optimal preheat start distance, km from charger
     *   T_end_opt       — predicted battery temp at charger arrival, C
     *   SOC_end_opt     — predicted SOC at charger arrival, %
     *   E_heat_opt      — total preheat energy consumption, kWh
     *   chrg_time_s   — predicted charging time to 80% SOC, s
     *
     * Hard Constraints:
     *   T_end_opt in [20, 25] C
     *   SOC_end_opt >= 10 %
     *
     * Multi-Objective Optimization (DESIGN YOUR OWN STRATEGY):
     *   - Minimize E_heat_opt      (preheat energy)
     *   - Minimize chrg_time_s   (charging time to 80% SOC)
     *
     *   There is no single "correct" answer. You decide the tradeoff.
     *   Hint: start_distance controls both objectives.
     *         Preheating earlier => higher energy but potentially shorter charge time.
     *         Preheating later  => lower energy but longer charge time.
     *         Find your own optimal balance.
     *
     * Hints:
     *   1. Build a forward simulation using the physical model above.
     *   2. Use binary search or grid search to explore start_distance.
     *   3. chrg_time_s: simulate charging from SOC_end_opt to SOC_TARGET (80%),
     *      using P_CHARGE_MAP to limit charging power at each (T, SOC) step.
     */
    Candidate optimum;
    int feasible_count = 0;
    const char *search_mode = "full-sweep";
    int optimization_status =
        mlp_guided_search(T_init, SOC_init * 100.0f, &optimum);
    if (optimization_status == 1) {
        search_mode = "mlp-guided";
    } else {
        optimization_status = optimize_preheat(T_init, SOC_init * 100.0f,
                                               &optimum, &feasible_count);
    }

    if (optimization_status < 0) {
        fprintf(stderr, "ERROR: invalid route or numerical simulation failure.\n");
        return 1;
    }
    if (optimization_status == 0) {
        printf("  WARNING: no feasible strategy; returning the least-violation candidate.\n");
    } else {
        printf("  Search mode: %s\n", search_mode);
        printf("  Best objective score: %.3f / 50\n", optimum.score);
    }

    float start_distance = (float)optimum.sim.start_distance_km;
    float T_end_opt      = (float)optimum.sim.end_temp_c;
    float SOC_end_opt    = (float)optimum.sim.end_soc_pct;
    float E_heat_opt     = (float)optimum.sim.heat_energy_kwh;
    float chrg_time_s    = (float)optimum.sim.charge_time_s;

    /* --- Step 3: Notify system of results --- */
    BattChrgPreHeatg_ntfPreHeatgStartDist(start_distance);
    BattChrgPreHeatg_ntfPreHeatgEndTemp(T_end_opt);
    BattChrgPreHeatg_ntfPreHeatgEndSOC(SOC_end_opt);
    BattChrgPreHeatg_ntfPreHeatgEnergy(E_heat_opt);

    /* --- Step 4: Verify results --- */
    printf("\n===== Result Verification =====\n");
    printf("  start_distance: notified=%.2f km\n", start_distance);
    printf("  T_end_opt:      notified=%.2f\n", T_end_opt);
    printf("  SOC_end_opt:    notified=%.2f\n", SOC_end_opt);
    printf("  E_heat_opt:     notified=%.4f\n", E_heat_opt);

    /* --- Step 5: Print results --- */
    printf("\n===== Results =====\n");
    printf("  start_distance = %.2f km  (begin preheat when %.2f km from charger)\n", start_distance, start_distance);
    printf("  T_end_opt      = %.2f C  [target: 20 ~ 25 C]\n", T_end_opt);
    printf("  SOC_end_opt    = %.2f %%  [min: 10 %%]\n", SOC_end_opt);
    printf("  E_heat_opt     = %.4f kWh  (%.0f kJ)\n", E_heat_opt, E_heat_opt * 3600.0f);
    printf("  chrg_time_s  = %.2f s  (to charge from %.1f%% to 80%%)\n", chrg_time_s, SOC_end_opt);

    /* --- Step 6: Constraint checks --- */
    printf("\n===== Constraint Checks =====\n");
    int pass = 1;
    if (T_end_opt >= T_OPT_LOW && T_end_opt <= T_OPT_HIGH) {
        printf("  [PASS] Temperature %.2f C in [%.0f, %.0f] C\n", T_end_opt, T_OPT_LOW, T_OPT_HIGH);
    } else {
        printf("  [FAIL] Temperature %.2f C NOT in [%.0f, %.0f] C\n", T_end_opt, T_OPT_LOW, T_OPT_HIGH);
        pass = 0;
    }
    if (SOC_end_opt >= SOC_MIN * 100.0f) {
        printf("  [PASS] SOC %.2f %% >= %.0f %%\n", SOC_end_opt, SOC_MIN * 100.0f);
    } else {
        printf("  [FAIL] SOC %.2f %% < %.0f %%\n", SOC_end_opt, SOC_MIN * 100.0f);
        pass = 0;
    }
    printf("  %s\n", pass ? "ALL CONSTRAINTS SATISFIED" : "SOME CONSTRAINTS VIOLATED");

    {
        RouteSummary route_stat;
        if (summarize_route(&route_stat)) {
            const double full_sims =
                ceil(route_stat.total_time_s / CANDIDATE_STEP_S) + 2.0;
            printf("\n  Physics simulations: %ld (full 1s sweep: ~%.0f, %.1f%%)\n",
                   g_sim_count, full_sims,
                   100.0 * (double)g_sim_count / full_sims);
        }
    }
    printf("\n===== Done =====\n");
    return 0;
}
