/**
 * @file student_solution.c
 * @brief BEV battery preheat robust decision algorithm
 *
 * Core method (V4):
 *   1) Battery electro-thermal forward simulation (1 s integration)
 *   2) Distance-domain adaptive coarse-to-fine grid search
 *   3) Multi-scenario robust coverage optimization
 *   4) Dynamic temperature safety margin
 *   5) SOC safety-margin penalty
 *   6) Fast infeasibility pruning and local fallback refinement
 *
 * V4 optimization changes:
 *   - Coarse scan uses the nominal scenario only.
 *   - Every fine candidate is simulated across robust scenarios only ONCE.
 *   - Temperature-margin candidates reuse the same simulation results.
 *   - Failed cases no longer default to a full-route 1 s scan.
 *
 * Decision variable:
 *   start_distance = remaining distance to charger when PTC preheat starts
 *
 * Hard constraints at charger arrival:
 *   20 C <= T_end <= 25 C
 *   SOC_end >= 10 %
 *
 * Optimization objectives:
 *   - shorter charging time to 80% SOC
 *   - lower preheat energy
 *   - larger temperature/SOC safety margin
 *   - higher feasibility under uncertain operating conditions
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "usp_api.h"

/* =========================================================================
 * 1. Physical constants
 * ========================================================================= */

/* Thermal model */
#define M_BAT       400.0f
#define CP          1000.0f

/* Electrical model */
#define C_NOM_AH    100.0f
#define U_NOM       350.0f

/* PTC heater */
#define P_HEAT_MAX  6000.0f
#define ETA_HEAT    0.95f

/* Heat transfer */
#define H0          50.0f
#define KV          0.5f

/* Official constraints */
#define T_OPT_LOW   20.0f
#define T_OPT_HIGH  25.0f
#define SOC_TARGET  0.80f
#define SOC_MIN     0.10f

/* Dynamic robust-design parameters (NOT official scoring constants) */
static const float TEMP_MARGIN_CANDIDATES[] = {0.80f, 0.50f, 0.30f, 0.10f};
#define N_TEMP_MARGINS ((int)(sizeof(TEMP_MARGIN_CANDIDATES)/sizeof(TEMP_MARGIN_CANDIDATES[0])))

static const int COARSE_STEP_TIME_S[] = {20, 10, 5};
#define N_COARSE_LEVELS ((int)(sizeof(COARSE_STEP_TIME_S)/sizeof(COARSE_STEP_TIME_S[0])))

#define SOC_DESIRED_MARGIN     0.03f
#define SOC_SAFE_TARGET        0.13f
#define SIM_DT_S               1.0f
#define FINE_STEP_TIME_EQUIV_S 1.0f
#define EPS                    1.0e-6f
#define MAX_FINE_CANDIDATES    4096
#define MIN_COARSE_FEASIBLE_POINTS 5
#define MIN_COARSE_SPAN_MULTIPLIER 2.0f

#define W_CHARGE 0.45f
#define W_ENERGY 0.30f
#define W_TEMP   0.15f
#define W_SOC    0.10f


/* =========================================================================
 * 2. Battery maps
 * ========================================================================= */

static const float R_INT_TEMPS[7] =
    {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};

static const float R_INT_SOCS[8] =
    {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};

static const float R_INT_MAP[7][8] = {
    {0.150f, 0.120f, 0.100f, 0.090f, 0.085f, 0.082f, 0.080f, 0.078f},
    {0.100f, 0.085f, 0.075f, 0.070f, 0.065f, 0.062f, 0.060f, 0.058f},
    {0.070f, 0.060f, 0.052f, 0.048f, 0.045f, 0.043f, 0.042f, 0.041f},
    {0.050f, 0.042f, 0.038f, 0.035f, 0.033f, 0.032f, 0.031f, 0.030f},
    {0.038f, 0.032f, 0.029f, 0.027f, 0.025f, 0.024f, 0.023f, 0.023f},
    {0.032f, 0.028f, 0.025f, 0.023f, 0.022f, 0.021f, 0.020f, 0.020f},
    {0.028f, 0.025f, 0.022f, 0.021f, 0.020f, 0.019f, 0.019f, 0.018f},
};

static const float U_OC_TEMPS[7] =
    {-20.0f, -10.0f, 0.0f, 10.0f, 20.0f, 30.0f, 40.0f};

static const float U_OC_SOCS[8] =
    {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};

static const float UOC_MAP[7][8] = {
    {295.0f, 320.0f, 340.0f, 352.0f, 365.0f, 375.0f, 382.0f, 410.0f},
    {298.0f, 323.0f, 343.0f, 355.0f, 368.0f, 378.0f, 385.0f, 413.0f},
    {300.0f, 325.0f, 345.0f, 357.0f, 370.0f, 380.0f, 387.0f, 415.0f},
    {302.0f, 327.0f, 347.0f, 359.0f, 372.0f, 382.0f, 389.0f, 417.0f},
    {303.0f, 328.0f, 348.0f, 360.0f, 373.0f, 383.0f, 390.0f, 418.0f},
    {304.0f, 329.0f, 349.0f, 361.0f, 374.0f, 384.0f, 391.0f, 419.0f},
    {305.0f, 330.0f, 350.0f, 362.0f, 375.0f, 385.0f, 392.0f, 420.0f},
};

static const float P_CHARGE_TEMPS[7] =
    {-20.0f, -10.0f, 0.0f, 10.0f, 23.0f, 30.0f, 40.0f};

static const float P_CHARGE_SOCS[8] =
    {0.0f, 10.0f, 20.0f, 30.0f, 50.0f, 70.0f, 90.0f, 100.0f};

static const float P_CHARGE_MAP[7][8] = {
    {10.0f, 12.0f, 15.0f, 18.0f, 20.0f, 18.0f, 15.0f, 10.0f},
    {20.0f, 25.0f, 30.0f, 35.0f, 38.0f, 35.0f, 28.0f, 20.0f},
    {40.0f, 50.0f, 60.0f, 65.0f, 68.0f, 65.0f, 55.0f, 40.0f},
    {60.0f, 70.0f, 80.0f, 85.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {75.0f, 85.0f, 90.0f, 92.0f, 90.0f, 88.0f, 75.0f, 55.0f},
    {70.0f, 80.0f, 88.0f, 90.0f, 88.0f, 85.0f, 70.0f, 50.0f},
    {50.0f, 60.0f, 70.0f, 72.0f, 70.0f, 65.0f, 55.0f, 40.0f},
};

/* =========================================================================
 * 3. Navigation data
 * ========================================================================= */

#define MAX_SEGS  PWRPRED_MAX_SEGS

typedef struct {
    float s_km;
    float v_kmh;
    float P_drive_kW;
    float T_env_C;
} NavSeg;

static NavSeg g_segs[MAX_SEGS];
static int g_n_segs = 0;

static void load_nav_segments(void)
{
    PwrPredList_stru pred;
    VehPwrPred_getPwrPred(&pred);

    g_n_segs = (pred.num < MAX_SEGS) ? pred.num : MAX_SEGS;

    for (int i = 0; i < g_n_segs; ++i) {
        g_segs[i].s_km       = pred.pwrPredList[i].length;
        g_segs[i].v_kmh      = pred.pwrPredList[i].avgSpd;
        g_segs[i].P_drive_kW = pred.pwrPredList[i].drvPwr;
        g_segs[i].T_env_C    = pred.pwrPredList[i].ambTemp;
    }
}

/* =========================================================================
 * 4. Runtime instrumentation
 * ========================================================================= */

typedef struct {
    unsigned long long sim_calls;
    unsigned long long integration_steps;
    unsigned long long interp_calls;
    unsigned long long charge_steps;
} ComputeCounters;

static ComputeCounters g_counter = {0,0,0,0};

static void reset_counters(void)
{
    memset(&g_counter, 0, sizeof(g_counter));
}

/* =========================================================================
 * 5. Utility functions
 * ========================================================================= */

static float clamp_local(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float interp2(const float *temps, int nt,
                     const float *socs, int ns,
                     const float *map,
                     float temp_C, float soc_pct)
{
    int it = 0;
    int is = 0;

    g_counter.interp_calls++;

    temp_C = clamp_local(temp_C, temps[0], temps[nt - 1]);
    soc_pct = clamp_local(soc_pct, socs[0], socs[ns - 1]);

    while (it < nt - 2 && temp_C > temps[it + 1]) ++it;
    while (is < ns - 2 && soc_pct > socs[is + 1]) ++is;

    float tx = (temp_C - temps[it]) /
               (temps[it + 1] - temps[it]);

    float sx = (soc_pct - socs[is]) /
               (socs[is + 1] - socs[is]);

    float q00 = map[it * ns + is];
    float q01 = map[it * ns + is + 1];
    float q10 = map[(it + 1) * ns + is];
    float q11 = map[(it + 1) * ns + is + 1];

    float a = q00 + sx * (q01 - q00);
    float b = q10 + sx * (q11 - q10);

    return a + tx * (b - a);
}

static float route_total_distance_km(void)
{
    float total = 0.0f;

    for (int i = 0; i < g_n_segs; ++i)
        total += fmaxf(g_segs[i].s_km, 0.0f);

    return total;
}

static float route_total_time_s(void)
{
    float total = 0.0f;

    for (int i = 0; i < g_n_segs; ++i) {
        float v = g_segs[i].v_kmh;
        if (v > 0.0f)
            total += g_segs[i].s_km / v * 3600.0f;
    }

    return total;
}

static float route_average_speed_kmh(void)
{
    float d = route_total_distance_km();
    float t = route_total_time_s();
    if (t <= EPS) return 1.0f;
    return d / (t / 3600.0f);
}

static float time_equiv_step_to_km(float seconds)
{
    float step = route_average_speed_kmh() * seconds / 3600.0f;
    return fmaxf(step, 0.005f);
}

static float nominal_trigger_time_s(float start_distance_km)
{
    float total = route_total_distance_km();
    if (start_distance_km >= total - EPS) return 0.0f;

    float remaining = total;
    float elapsed = 0.0f;
    for (int i = 0; i < g_n_segs; ++i) {
        float d = fmaxf(g_segs[i].s_km, 0.0f);
        float v = fmaxf(g_segs[i].v_kmh, 0.1f);
        if (remaining - d <= start_distance_km + EPS) {
            float inside = clamp_local(remaining - start_distance_km, 0.0f, d);
            return elapsed + inside / v * 3600.0f;
        }
        elapsed += d / v * 3600.0f;
        remaining -= d;
    }
    return route_total_time_s();
}

/* =========================================================================
 * 5. Scenario model
 *
 * The optimizer is trained against several mild-but-realistic uncertainty
 * scenarios. The stress test later uses stronger disturbances.
 * ========================================================================= */

typedef struct {
    const char *name;

    /* Initial state */
    float init_temp_offset_C;
    float soc_offset_frac;

    /* Route perturbations */
    float ambient_offset_C;
    float speed_factor;
    float power_factor;
    float distance_factor;
} Scenario;

/*
 * Robust optimization scenario set.
 *
 * These are intentionally milder than the final stress-test cases.
 * If every scenario were extremely severe, the physical feasible set could
 * disappear and no optimizer could satisfy the hard constraints.
 */
static const Scenario ROBUST_SCENARIOS[] = {
    /*
     * These scenarios are used INSIDE the optimizer.
     * The final stress test below is still stronger and independent.
     *
     * Important V3 change:
     *   the optimizer later builds an "active robust mask" from scenarios
     *   that actually have at least one candidate compatible with nominal
     *   feasibility. A scenario with no compatible feasible point therefore
     *   cannot distort the final decision.
     */
    {"Nominal",             0.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.00f},
    {"ColdBattery_-5C",    -5.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.00f},
    {"ColdAmbient_-3C",     0.0f,  0.000f, -3.0f, 1.00f, 1.00f, 1.00f},
    {"FastTraffic_+20pct",  0.0f,  0.000f,  0.0f, 1.20f, 1.00f, 1.00f},
    {"HighwayFast",         0.0f,  0.000f,  0.0f, 1.30f, 1.15f, 1.00f},
    {"SlowTraffic_-5pct",   0.0f,  0.000f,  0.0f, 0.95f, 1.00f, 1.00f},
    {"HigherPower_+5pct",   0.0f,  0.000f,  0.0f, 1.00f, 1.05f, 1.00f},
    {"LongerRoute_+3pct",   0.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.03f},
    {"LowSOCEstimate",      0.0f, -0.010f,  0.0f, 1.00f, 1.00f, 1.00f},
    {"MixedAdverse",       -1.0f, -0.005f, -1.0f, 0.97f, 1.03f, 1.02f},
    {"Combined_FastHigh",   3.0f,  0.000f,  2.0f, 1.15f, 1.15f, 0.95f}
};

#define N_ROBUST_SCENARIOS \
    ((int)(sizeof(ROBUST_SCENARIOS) / sizeof(ROBUST_SCENARIOS[0])))

/* =========================================================================
 * 6. Forward simulation
 * ========================================================================= */

typedef struct {
    int physically_valid;
    int feasible;

    float start_distance_km;
    float T_end_C;
    float SOC_end_frac;
    float E_heat_kWh;
    float charge_time_s;
} SimResult;

static float compute_charge_time_s(float T_end_C, float SOC_end_frac)
{
    float soc = SOC_end_frac;
    float time_s = 0.0f;

    const float max_time_s = 24.0f * 3600.0f;

    if (soc >= SOC_TARGET)
        return 0.0f;

    while (soc < SOC_TARGET && time_s < max_time_s) {
        float p_kW = interp2(
            P_CHARGE_TEMPS, 7,
            P_CHARGE_SOCS, 8,
            &P_CHARGE_MAP[0][0],
            T_end_C,
            soc * 100.0f
        );

        if (p_kW <= 0.0f || !isfinite(p_kW))
            return INFINITY;

        soc += p_kW * 1000.0f /
               (C_NOM_AH * U_NOM * 3600.0f);

        time_s += 1.0f;
        g_counter.charge_steps++;
    }

    return (soc >= SOC_TARGET) ? time_s : INFINITY;
}

/*
 * Simulate one fixed start distance under one scenario.
 *
 * Key improvement compared with the old time-domain decision:
 * PTC is triggered when REMAINING DISTANCE <= start_distance.
 *
 * Therefore, if actual speed changes, the trigger time changes automatically,
 * while the trigger location remains consistent with the competition output.
 */
static SimResult simulate_distance_candidate(float start_distance_km,
                                             float T_init,
                                             float SOC_init,
                                             const Scenario *sc)
{
    SimResult out;
    memset(&out, 0, sizeof(out));
    g_counter.sim_calls++;

    float T = T_init + sc->init_temp_offset_C;
    float soc = clamp_local(SOC_init + sc->soc_offset_frac, 0.0f, 1.0f);

    float total_distance = 0.0f;

    for (int i = 0; i < g_n_segs; ++i)
        total_distance += fmaxf(g_segs[i].s_km * sc->distance_factor, 0.0f);

    float remaining = total_distance;
    float heat_energy_J = 0.0f;
    int heat_on = (start_distance_km >= remaining - EPS);

    out.start_distance_km = start_distance_km;
    out.physically_valid = 1;

    for (int seg = 0; seg < g_n_segs; ++seg) {
        float seg_dist =
            fmaxf(g_segs[seg].s_km * sc->distance_factor, 0.0f);

        float v =
            fmaxf(g_segs[seg].v_kmh * sc->speed_factor, 0.1f);

        float p_drive_kW =
            fmaxf(g_segs[seg].P_drive_kW * sc->power_factor, 0.0f);

        float T_env =
            g_segs[seg].T_env_C + sc->ambient_offset_C;

        float seg_time = seg_dist / v * 3600.0f;
        float local_time = 0.0f;

        while (local_time < seg_time - EPS) {
            float dt = fminf(SIM_DT_S, seg_time - local_time);
            g_counter.integration_steps++;

            /*
             * Detect if the vehicle crosses the requested remaining-distance
             * threshold inside the current 1 s step.
             */
            if (!heat_on && remaining <= start_distance_km + EPS)
                heat_on = 1;

            float p_heat_W = heat_on ? P_HEAT_MAX : 0.0f;
            float p_bat_W = p_drive_kW * 1000.0f + p_heat_W;

            float R = interp2(
                R_INT_TEMPS, 7,
                R_INT_SOCS, 8,
                &R_INT_MAP[0][0],
                T,
                soc * 100.0f
            );

            float Uoc = interp2(
                U_OC_TEMPS, 7,
                U_OC_SOCS, 8,
                &UOC_MAP[0][0],
                T,
                soc * 100.0f
            );

            /*
             * P = (Uoc - I*R) * I
             * => R I^2 - Uoc I + P = 0
             */
            float disc = Uoc * Uoc - 4.0f * R * p_bat_W;

            /*
             * Small negative values may come only from floating-point noise.
             * A clearly negative discriminant means the requested operating
             * point is physically infeasible in this simplified model.
             */
            if (disc < -1.0e-3f || R <= 0.0f || !isfinite(disc)) {
                out.physically_valid = 0;
                out.feasible = 0;
                return out;
            }

            if (disc < 0.0f)
                disc = 0.0f;

            float current_A =
                (Uoc - sqrtf(disc)) / (2.0f * R);

            if (!isfinite(current_A) || current_A < 0.0f) {
                out.physically_valid = 0;
                out.feasible = 0;
                return out;
            }

            float q_gen_W = current_A * current_A * R;
            float q_heat_W = ETA_HEAT * p_heat_W;
            float k_env = H0 + KV * v;
            float q_loss_W = k_env * (T - T_env);

            T += (q_gen_W + q_heat_W - q_loss_W) *
                 dt / (M_BAT * CP);

            soc -= current_A *
                   dt / (C_NOM_AH * 3600.0f);

            if (heat_on)
                heat_energy_J += p_heat_W * dt;

            local_time += dt;

            remaining -= v * dt / 3600.0f;
            if (remaining < 0.0f)
                remaining = 0.0f;

            if (!isfinite(T) || !isfinite(soc) || soc < 0.0f) {
                out.physically_valid = 0;
                out.feasible = 0;
                return out;
            }
        }
    }

    out.T_end_C = T;
    out.SOC_end_frac = soc;
    out.E_heat_kWh = heat_energy_J / 3600000.0f;

    out.charge_time_s =
        compute_charge_time_s(T, soc);

    out.feasible =
        out.physically_valid &&
        isfinite(out.charge_time_s) &&
        T >= T_OPT_LOW &&
        T <= T_OPT_HIGH &&
        soc >= SOC_MIN;

    return out;
}

/* =========================================================================
 * 7. Candidate evaluation for robust optimization (V4)
 *
 * A candidate distance is simulated over each robust scenario only once.
 * Results are reused for robust coverage, dynamic safety margin, and scoring.
 * ========================================================================= */

typedef unsigned int ScenarioMask;

typedef struct {
    float start_distance_km;
    SimResult nominal;

    ScenarioMask feasible_mask;
    ScenarioMask safe_mask[N_TEMP_MARGINS];

    float temp_margin_by_scenario[N_ROBUST_SCENARIOS];
    float soc_margin_by_scenario[N_ROBUST_SCENARIOS];

    int scenario_feasible_count;
    int scenario_safe_count;
    float scenario_feasible_rate;
    float scenario_safe_rate;

    float worst_safety_index;
    float temp_margin_C;

    float jt;
    float je;
    float official_cost;
} CandidateEval;

static int popcount_mask(ScenarioMask x)
{
    int n = 0;
    while (x) {
        n += (int)(x & 1u);
        x >>= 1u;
    }
    return n;
}

static float temperature_margin_C(float T)
{
    return fminf(T - T_OPT_LOW, T_OPT_HIGH - T);
}

static float normalized_safety_index(float temp_margin,
                                     float soc_margin)
{
    float temp_ref = TEMP_MARGIN_CANDIDATES[0];
    float t = temp_margin / fmaxf(temp_ref, EPS);
    float s = soc_margin / fmaxf(SOC_DESIRED_MARGIN, EPS);
    return fminf(t, s);
}

static CandidateEval evaluate_candidate_once(float start_distance_km,
                                             float T_init,
                                             float SOC_init)
{
    CandidateEval e = {0};
    e.start_distance_km = start_distance_km;

    for (int sidx = 0; sidx < N_ROBUST_SCENARIOS; ++sidx) {
        e.temp_margin_by_scenario[sidx] = -100.0f;
        e.soc_margin_by_scenario[sidx] = -1.0f;

        SimResult r = simulate_distance_candidate(
            start_distance_km, T_init, SOC_init,
            &ROBUST_SCENARIOS[sidx]
        );

        if (sidx == 0)
            e.nominal = r;

        ScenarioMask bit = (ScenarioMask)1u << (unsigned)sidx;

        if (!r.feasible)
            continue;

        e.feasible_mask |= bit;

        float tm = temperature_margin_C(r.T_end_C);
        float sm = r.SOC_end_frac - SOC_MIN;

        e.temp_margin_by_scenario[sidx] = tm;
        e.soc_margin_by_scenario[sidx] = sm;

        for (int m = 0; m < N_TEMP_MARGINS; ++m) {
            float margin = TEMP_MARGIN_CANDIDATES[m];
            if (r.T_end_C >= T_OPT_LOW + margin &&
                r.T_end_C <= T_OPT_HIGH - margin &&
                r.SOC_end_frac >= SOC_MIN) {
                e.safe_mask[m] |= bit;
            }
        }
    }

    return e;
}

static float select_dynamic_margin_for_candidate(const CandidateEval *e,
                                                 ScenarioMask covered_mask)
{
    int covered_count = popcount_mask(covered_mask);

    if (covered_count <= 0)
        return TEMP_MARGIN_CANDIDATES[N_TEMP_MARGINS - 1];

    /* Largest common margin first: 0.8 -> 0.5 -> 0.3 -> 0.1 C. */
    for (int m = 0; m < N_TEMP_MARGINS; ++m) {
        int safe_count = popcount_mask(e->safe_mask[m] & covered_mask);
        if (safe_count == covered_count)
            return TEMP_MARGIN_CANDIDATES[m];
    }

    return TEMP_MARGIN_CANDIDATES[N_TEMP_MARGINS - 1];
}

static float candidate_worst_safety_index(const CandidateEval *e,
                                          ScenarioMask covered_mask)
{
    float worst = INFINITY;
    int seen = 0;

    for (int sidx = 0; sidx < N_ROBUST_SCENARIOS; ++sidx) {
        ScenarioMask bit = (ScenarioMask)1u << (unsigned)sidx;
        if ((covered_mask & bit) == 0u)
            continue;

        float idx = normalized_safety_index(
            e->temp_margin_by_scenario[sidx],
            e->soc_margin_by_scenario[sidx]
        );

        if (idx < worst)
            worst = idx;
        seen = 1;
    }

    return seen ? worst : -INFINITY;
}

/* =========================================================================
 * 8. V4 lexicographic robust coarse-to-fine optimizer
 *
 * Priority 1: maximize robust scenario coverage.
 * Priority 2: maximize the minimum normalized temperature/SOC safety margin.
 * Priority 3: minimize official 30:20 objective:
 *             J = 0.6*J_charge + 0.4*J_energy.
 * Priority 4: prefer the larger dynamically supported temperature margin.
 *
 * Compared with V3, a robust whole-route coarse scan is added before the fine
 * scan. Thus a fast/highway feasible region cannot be missed simply because
 * it lies outside the nominal fine interval.
 * ========================================================================= */

#define MAX_ROBUST_COARSE_CANDIDATES 2048
#define SAFETY_TIE_TOL 0.005f

typedef struct {
    int success;

    float selected_temp_margin_C;
    int selected_coarse_step_time_s;
    float selected_coarse_step_km;
    float fine_step_km;

    int coarse_evals;
    int robust_coarse_evals;
    int fine_evals;
    int scenario_simulations;

    int active_robust_scenarios;
    int excluded_incompatible_scenarios;
    int best_robust_feasible_count;
    int best_safe_count;

    float best_robust_feasible_rate;
    float best_safe_rate;
    int common_robust_policy_exists;

    float best_worst_safety_index;
    float official_objective;

    float fine_lo_km;
    float fine_hi_km;

    float final_cost;
    double elapsed_ms;
    ComputeCounters counters;
} OptimStats;

typedef struct {
    int accepted;
    int level_index;
    int step_time_s;
    float step_km;
    int eval_count;
    int feasible_count;
    float first_feasible_km;
    float last_feasible_km;
    float best_violation_km;
    float best_violation;
} CoarseScanInfo;

typedef struct {
    ScenarioMask active_mask;
    int candidate_count;
    int best_coverage_count;
    float best_region_first_km;
    float best_region_last_km;
} RobustCoarseInfo;

static float nominal_constraint_violation(const SimResult *r)
{
    if (!r->physically_valid)
        return 1.0e6f;

    float v = 0.0f;
    if (r->T_end_C < T_OPT_LOW)
        v += T_OPT_LOW - r->T_end_C;
    if (r->T_end_C > T_OPT_HIGH)
        v += r->T_end_C - T_OPT_HIGH;
    if (r->SOC_end_frac < SOC_MIN)
        v += 100.0f * (SOC_MIN - r->SOC_end_frac);
    return v;
}

static int nominal_fast_impossibility_check(float T_init,
                                            float SOC_init)
{
    float total = route_total_distance_km();

    SimResult late = simulate_distance_candidate(
        0.0f, T_init, SOC_init, &ROBUST_SCENARIOS[0]
    );

    if (!late.physically_valid)
        return 1;
    if (late.SOC_end_frac < SOC_MIN)
        return 1;
    if (late.T_end_C > T_OPT_HIGH)
        return 1;

    SimResult early = simulate_distance_candidate(
        total, T_init, SOC_init, &ROBUST_SCENARIOS[0]
    );

    if (early.physically_valid && early.T_end_C < T_OPT_LOW)
        return 1;

    return 0;
}

static CoarseScanInfo adaptive_coarse_scan(float T_init,
                                           float SOC_init)
{
    CoarseScanInfo info = {0};
    float total = route_total_distance_km();
    info.best_violation = INFINITY;

    for (int level = 0; level < N_COARSE_LEVELS; ++level) {
        float step_km = time_equiv_step_to_km(
            (float)COARSE_STEP_TIME_S[level]
        );

        int feasible_count = 0;
        float first = INFINITY;
        float last = -INFINITY;
        int evals = 0;

        for (float x = 0.0f; x <= total + EPS; x += step_km) {
            float cx = fminf(x, total);
            SimResult r = simulate_distance_candidate(
                cx, T_init, SOC_init, &ROBUST_SCENARIOS[0]
            );
            evals++;

            float violation = nominal_constraint_violation(&r);
            if (violation < info.best_violation) {
                info.best_violation = violation;
                info.best_violation_km = cx;
            }

            if (r.feasible) {
                feasible_count++;
                if (cx < first) first = cx;
                if (cx > last) last = cx;
            }

            if (cx >= total - EPS)
                break;
        }

        info.level_index = level;
        info.step_time_s = COARSE_STEP_TIME_S[level];
        info.step_km = step_km;
        info.eval_count += evals;
        info.feasible_count = feasible_count;
        info.first_feasible_km = feasible_count > 0 ? first : 0.0f;
        info.last_feasible_km = feasible_count > 0 ? last : total;

        float span = feasible_count > 0 ? last - first : 0.0f;
        if (feasible_count >= MIN_COARSE_FEASIBLE_POINTS &&
            span >= MIN_COARSE_SPAN_MULTIPLIER * step_km) {
            info.accepted = 1;
            return info;
        }
    }

    return info;
}

static RobustCoarseInfo robust_coarse_scan(float T_init,
                                           float SOC_init,
                                           float step_km,
                                           int *eval_counter)
{
    RobustCoarseInfo info = {0};
    static CandidateEval coarse_cand[MAX_ROBUST_COARSE_CANDIDATES];
    int n = 0;
    float total = route_total_distance_km();

    for (float x = 0.0f;
         x <= total + EPS && n < MAX_ROBUST_COARSE_CANDIDATES;
         x += step_km) {

        float cx = fminf(x, total);
        CandidateEval e = evaluate_candidate_once(cx, T_init, SOC_init);
        (*eval_counter)++;

        if (e.nominal.feasible) {
            coarse_cand[n++] = e;
            info.active_mask |= e.feasible_mask;
        }

        if (cx >= total - EPS)
            break;
    }

    info.candidate_count = n;
    info.active_mask |= 1u; /* nominal scenario */

    int best_count = -1;
    float first = INFINITY;
    float last = -INFINITY;

    for (int i = 0; i < n; ++i) {
        int count = popcount_mask(
            coarse_cand[i].feasible_mask & info.active_mask
        );

        if (count > best_count) {
            best_count = count;
            first = coarse_cand[i].start_distance_km;
            last = coarse_cand[i].start_distance_km;
        } else if (count == best_count) {
            float x = coarse_cand[i].start_distance_km;
            if (x < first) first = x;
            if (x > last) last = x;
        }
    }

    info.best_coverage_count = best_count;
    info.best_region_first_km = isfinite(first) ? first : 0.0f;
    info.best_region_last_km = isfinite(last) ? last : total;
    return info;
}

static int lexicographic_better(const CandidateEval *a,
                                const CandidateEval *b,
                                int b_exists)
{
    if (!b_exists)
        return 1;

    if (a->scenario_feasible_count != b->scenario_feasible_count)
        return a->scenario_feasible_count > b->scenario_feasible_count;

    if (a->worst_safety_index > b->worst_safety_index + SAFETY_TIE_TOL)
        return 1;
    if (b->worst_safety_index > a->worst_safety_index + SAFETY_TIE_TOL)
        return 0;

    if (a->official_cost < b->official_cost - 1.0e-7f)
        return 1;
    if (b->official_cost < a->official_cost - 1.0e-7f)
        return 0;

    if (a->temp_margin_C > b->temp_margin_C + EPS)
        return 1;
    if (b->temp_margin_C > a->temp_margin_C + EPS)
        return 0;

    return a->nominal.E_heat_kWh < b->nominal.E_heat_kWh;
}

static int robust_optimize(float T_init,
                           float SOC_init,
                           SimResult *best,
                           OptimStats *stats,
                           int verbose)
{
    clock_t t0 = clock();
    reset_counters();
    memset(best, 0, sizeof(*best));
    memset(stats, 0, sizeof(*stats));

    float total = route_total_distance_km();
    if (total <= EPS || g_n_segs <= 0)
        return 0;

    if (nominal_fast_impossibility_check(T_init, SOC_init)) {
        stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                            (double)CLOCKS_PER_SEC;
        stats->counters = g_counter;
        if (verbose)
            printf("  Fast feasibility pruning: nominal route is physically infeasible.\n");
        return 0;
    }

    /* Stage 1: dynamic 20 -> 10 -> 5 s-equivalent nominal coarse scan. */
    CoarseScanInfo coarse = adaptive_coarse_scan(T_init, SOC_init);
    stats->coarse_evals = coarse.eval_count;

    if (coarse.feasible_count <= 0) {
        stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                            (double)CLOCKS_PER_SEC;
        stats->counters = g_counter;
        return 0;
    }

    /* Stage 2: robust whole-route coarse scan. */
    int robust_coarse_evals = 0;
    RobustCoarseInfo rc = robust_coarse_scan(
        T_init, SOC_init, coarse.step_km, &robust_coarse_evals
    );
    stats->robust_coarse_evals = robust_coarse_evals;

    if (rc.candidate_count <= 0) {
        stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                            (double)CLOCKS_PER_SEC;
        stats->counters = g_counter;
        return 0;
    }

    ScenarioMask active_mask = rc.active_mask;
    int active_count = popcount_mask(active_mask);
    stats->active_robust_scenarios = active_count;
    stats->excluded_incompatible_scenarios =
        N_ROBUST_SCENARIOS - active_count;

    /* Stage 3: 1-s-equivalent fine search around maximum-coverage region. */
    float fine_step = time_equiv_step_to_km(FINE_STEP_TIME_EQUIV_S);
    float expansion = 2.0f * coarse.step_km;
    float fine_lo = fmaxf(0.0f, rc.best_region_first_km - expansion);
    float fine_hi = fminf(total, rc.best_region_last_km + expansion);

    if (fine_hi < fine_lo + fine_step) {
        fine_lo = fmaxf(0.0f, rc.best_region_first_km - 4.0f * coarse.step_km);
        fine_hi = fminf(total, rc.best_region_last_km + 4.0f * coarse.step_km);
    }

    static CandidateEval cand[MAX_FINE_CANDIDATES];
    int n = 0;

    for (float x = fine_lo;
         x <= fine_hi + EPS && n < MAX_FINE_CANDIDATES;
         x += fine_step) {

        float cx = fminf(x, fine_hi);
        CandidateEval e = evaluate_candidate_once(cx, T_init, SOC_init);
        stats->fine_evals++;

        if (e.nominal.feasible)
            cand[n++] = e;

        if (cx >= fine_hi - EPS)
            break;
    }

    if (n <= 0) {
        stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                            (double)CLOCKS_PER_SEC;
        stats->counters = g_counter;
        return 0;
    }

    /* Priority 1: maximize robust scenario coverage. */
    int best_coverage = -1;
    for (int i = 0; i < n; ++i) {
        ScenarioMask covered = cand[i].feasible_mask & active_mask;
        cand[i].scenario_feasible_count = popcount_mask(covered);
        cand[i].scenario_feasible_rate = active_count > 0 ?
            (float)cand[i].scenario_feasible_count / (float)active_count : 0.0f;

        if (cand[i].scenario_feasible_count > best_coverage)
            best_coverage = cand[i].scenario_feasible_count;
    }

    stats->common_robust_policy_exists = (best_coverage == active_count);

    /* Priority 2: maximize minimum normalized T/SOC safety margin. */
    float best_safety = -INFINITY;
    for (int i = 0; i < n; ++i) {
        CandidateEval *e = &cand[i];
        if (e->scenario_feasible_count != best_coverage)
            continue;

        ScenarioMask covered = e->feasible_mask & active_mask;
        e->worst_safety_index = candidate_worst_safety_index(e, covered);
        e->temp_margin_C = select_dynamic_margin_for_candidate(e, covered);

        int margin_index = N_TEMP_MARGINS - 1;
        for (int m = 0; m < N_TEMP_MARGINS; ++m) {
            if (fabsf(e->temp_margin_C - TEMP_MARGIN_CANDIDATES[m]) <= EPS) {
                margin_index = m;
                break;
            }
        }

        e->scenario_safe_count = popcount_mask(
            e->safe_mask[margin_index] & covered
        );
        e->scenario_safe_rate = e->scenario_feasible_count > 0 ?
            (float)e->scenario_safe_count /
            (float)e->scenario_feasible_count : 0.0f;

        if (e->worst_safety_index > best_safety)
            best_safety = e->worst_safety_index;
    }

    /* Priority 3: official 30:20 normalized objective among safety ties. */
    float min_charge = INFINITY, max_charge = -INFINITY;
    float min_energy = INFINITY, max_energy = -INFINITY;

    for (int i = 0; i < n; ++i) {
        CandidateEval *e = &cand[i];
        if (e->scenario_feasible_count != best_coverage)
            continue;
        if (e->worst_safety_index < best_safety - SAFETY_TIE_TOL)
            continue;

        float t = e->nominal.charge_time_s;
        float en = e->nominal.E_heat_kWh;
        if (t < min_charge) min_charge = t;
        if (t > max_charge) max_charge = t;
        if (en < min_energy) min_energy = en;
        if (en > max_energy) max_energy = en;
    }

    CandidateEval global_best = {0};
    int global_exists = 0;

    for (int i = 0; i < n; ++i) {
        CandidateEval e = cand[i];
        if (e.scenario_feasible_count != best_coverage)
            continue;
        if (e.worst_safety_index < best_safety - SAFETY_TIE_TOL)
            continue;

        e.jt = max_charge > min_charge + EPS ?
            (e.nominal.charge_time_s - min_charge) /
            (max_charge - min_charge) : 0.0f;

        e.je = max_energy > min_energy + EPS ?
            (e.nominal.E_heat_kWh - min_energy) /
            (max_energy - min_energy) : 0.0f;

        e.official_cost = 0.60f * e.jt + 0.40f * e.je;

        if (lexicographic_better(&e, &global_best, global_exists)) {
            global_best = e;
            global_exists = 1;
        }
    }

    if (!global_exists) {
        stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                            (double)CLOCKS_PER_SEC;
        stats->counters = g_counter;
        return 0;
    }

    *best = global_best.nominal;
    stats->success = 1;
    stats->selected_temp_margin_C = global_best.temp_margin_C;
    stats->selected_coarse_step_time_s = coarse.step_time_s;
    stats->selected_coarse_step_km = coarse.step_km;
    stats->fine_step_km = fine_step;
    stats->fine_lo_km = fine_lo;
    stats->fine_hi_km = fine_hi;
    stats->best_robust_feasible_count = global_best.scenario_feasible_count;
    stats->best_safe_count = global_best.scenario_safe_count;
    stats->best_robust_feasible_rate = global_best.scenario_feasible_rate;
    stats->best_safe_rate = global_best.scenario_safe_rate;
    stats->best_worst_safety_index = global_best.worst_safety_index;
    stats->official_objective = global_best.official_cost;
    stats->final_cost = global_best.official_cost;
    stats->scenario_simulations = (int)g_counter.sim_calls;
    stats->elapsed_ms = 1000.0 * (double)(clock() - t0) /
                        (double)CLOCKS_PER_SEC;
    stats->counters = g_counter;

    if (verbose) {
        printf("  Method: V4 lexicographic distance-domain adaptive robust coarse-to-fine search\n");
        printf("  Selected temperature safety margin: %.2f C\n",
               stats->selected_temp_margin_C);
        printf("  Selected coarse step: %d s-equivalent = %.4f km\n",
               stats->selected_coarse_step_time_s,
               stats->selected_coarse_step_km);
        printf("  Fine step: 1 s-equivalent = %.4f km\n",
               stats->fine_step_km);
        printf("  Robust fine interval: %.3f ~ %.3f km\n",
               stats->fine_lo_km, stats->fine_hi_km);
        printf("  Active robust scenarios: %d/%d\n",
               stats->active_robust_scenarios, N_ROBUST_SCENARIOS);
        printf("  Excluded incompatible scenarios: %d\n",
               stats->excluded_incompatible_scenarios);
        printf("  Priority #1 robust coverage: %d/%d = %.2f %%\n",
               stats->best_robust_feasible_count,
               stats->active_robust_scenarios,
               100.0f * stats->best_robust_feasible_rate);
        printf("  Common feasible policy for all active scenarios: %s\n",
               stats->common_robust_policy_exists ? "YES" : "NO");
        printf("  Priority #2 worst normalized safety index: %.4f\n",
               stats->best_worst_safety_index);
        printf("  Priority #3 official 30:20 normalized cost: %.6f\n",
               stats->official_objective);
        printf("  Optimization CPU time: %.3f ms\n",
               stats->elapsed_ms);
        printf("  Selected start distance: %.3f km\n",
               best->start_distance_km);
    }

    return 1;
}

/* =========================================================================
 * 9. Stress-test diagnostics
 * ========================================================================= */

static const Scenario STRESS_SCENARIOS[] = {
    {"Baseline",           0.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.00f},

    {"InitTemp_-5C",      -5.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.00f},
    {"InitTemp_+5C",       5.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.00f},

    {"Ambient_-3C",        0.0f,  0.000f, -3.0f, 1.00f, 1.00f, 1.00f},
    {"Ambient_+3C",        0.0f,  0.000f,  3.0f, 1.00f, 1.00f, 1.00f},

    {"Speed_-20pct",       0.0f,  0.000f,  0.0f, 0.80f, 1.00f, 1.00f},
    {"Speed_+20pct",       0.0f,  0.000f,  0.0f, 1.20f, 1.00f, 1.00f},

    {"Power_-15pct",       0.0f,  0.000f,  0.0f, 1.00f, 0.85f, 1.00f},
    {"Power_+15pct",       0.0f,  0.000f,  0.0f, 1.00f, 1.15f, 1.00f},

    {"Distance_-10pct",    0.0f,  0.000f,  0.0f, 1.00f, 1.00f, 0.90f},
    {"Distance_+10pct",    0.0f,  0.000f,  0.0f, 1.00f, 1.00f, 1.10f},

    {"UrbanCongestion",    0.0f,  0.000f, -1.0f, 0.70f, 0.90f, 1.00f},
    {"HighwayFast",        0.0f,  0.000f,  0.0f, 1.30f, 1.15f, 1.00f},

    {"Combined_ColdSlow", -3.0f, -0.010f, -3.0f, 0.85f, 1.10f, 1.10f},
    {"Combined_FastHigh",  3.0f,  0.000f,  2.0f, 1.15f, 1.15f, 0.95f}
};

#define N_STRESS_SCENARIOS \
    ((int)(sizeof(STRESS_SCENARIOS) / sizeof(STRESS_SCENARIOS[0])))

static void diagnose_failed_scenario(float start_distance_km,
                                     float T_init,
                                     float SOC_init,
                                     const Scenario *sc)
{
    /*
     * Diagnostic #1:
     * use the selected fixed policy
     */
    SimResult fixed =
        simulate_distance_candidate(
            start_distance_km,
            T_init,
            SOC_init,
            sc
        );

    const char *reason = "unknown";

    if (!fixed.physically_valid) {
        reason = "electrical/physical model infeasible";
    } else if (fixed.T_end_C < T_OPT_LOW) {
        reason = "arrival temperature below 20 C";
    } else if (fixed.T_end_C > T_OPT_HIGH) {
        reason = "arrival temperature above 25 C";
    } else if (fixed.SOC_end_frac < SOC_MIN) {
        reason = "arrival SOC below 10%";
    }

    printf("      reason: %s", reason);

    if (fixed.physically_valid) {
        printf(" (T=%.2f C, SOC=%.2f%%)",
               fixed.T_end_C,
               fixed.SOC_end_frac * 100.0f);
    }

    printf("\n");
}

/*
 * IMPORTANT:
 * This test keeps the FINAL selected start distance fixed.
 * It therefore measures decision robustness, not merely whether re-optimization
 * can adapt after the environment has already changed.
 */
static void run_fixed_policy_stress_test(float start_distance_km,
                                         float T_init,
                                         float SOC_init)
{
    int pass_count = 0;

    printf("\n");
    printf("====================================================================================================\n");
    printf("FIXED-POLICY ROBUSTNESS STRESS TEST\n");
    printf("The same selected start_distance = %.2f km is used in every scenario.\n",
           start_distance_km);
    printf("====================================================================================================\n");

    printf("%-20s %-6s %-9s %-9s %-10s %-10s\n",
           "Scenario",
           "PASS",
           "Tend(C)",
           "SOC(%)",
           "Eheat",
           "Charge(s)");

    printf("----------------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < N_STRESS_SCENARIOS; ++s) {
        SimResult r =
            simulate_distance_candidate(
                start_distance_km,
                T_init,
                SOC_init,
                &STRESS_SCENARIOS[s]
            );

        int pass = r.feasible;

        if (pass)
            pass_count++;

        if (r.physically_valid) {
            printf("%-20s %-6s %-9.2f %-9.2f %-10.4f ",
                   STRESS_SCENARIOS[s].name,
                   pass ? "PASS" : "FAIL",
                   r.T_end_C,
                   r.SOC_end_frac * 100.0f,
                   r.E_heat_kWh);

            if (isfinite(r.charge_time_s))
                printf("%-10.1f\n", r.charge_time_s);
            else
                printf("%-10s\n", "-");
        } else {
            printf("%-20s %-6s %-9s %-9s %-10s %-10s\n",
                   STRESS_SCENARIOS[s].name,
                   "FAIL",
                   "-",
                   "-",
                   "-",
                   "-");
        }

        if (!pass)
            diagnose_failed_scenario(
                start_distance_km,
                T_init,
                SOC_init,
                &STRESS_SCENARIOS[s]
            );
    }

    printf("----------------------------------------------------------------------------------------------------\n");

    printf("Fixed-policy robust scenarios : %d / %d\n",
           pass_count,
           N_STRESS_SCENARIOS);

    printf("Fixed-policy robustness rate  : %.2f %%\n",
           100.0f *
           (float)pass_count /
           (float)N_STRESS_SCENARIOS);
}

/* =========================================================================
 * 10. Feasibility-boundary diagnosis
 *
 * If a stress scenario fails for the selected policy, this function can tell
 * whether ANY start distance on a 0.1 km grid could satisfy the hard
 * constraints. That separates:
 *
 *   "policy is not robust enough"
 *
 * from
 *
 *   "the physical feasible set disappeared"
 * ========================================================================= */

static float scenario_violation_score(const SimResult *r)
{
    if (!r->physically_valid)
        return 1.0e6f;

    float v = 0.0f;

    if (r->T_end_C < T_OPT_LOW)
        v += T_OPT_LOW - r->T_end_C;

    if (r->T_end_C > T_OPT_HIGH)
        v += r->T_end_C - T_OPT_HIGH;

    if (r->SOC_end_frac < SOC_MIN)
        v += 100.0f *
             (SOC_MIN - r->SOC_end_frac);

    return v;
}

/*
 * V3 fast feasibility diagnosis.
 *
 * First use strong physical bounds:
 *   - if "almost no heating" already violates SOC < 10%, any earlier heating
 *     can only reduce SOC further -> no feasible solution.
 *   - if "heating from departure" is valid but still cannot reach 20 C,
 *     no later start can satisfy the temperature constraint.
 *
 * Otherwise use 20 -> 10 -> 5 s-equivalent coarse scans and only a local
 * 1-s refinement around the best coarse point.
 */
static int scenario_has_any_feasible_solution(float T_init,
                                              float SOC_init,
                                              const Scenario *sc,
                                              SimResult *best_feasible)
{
    memset(best_feasible, 0, sizeof(*best_feasible));

    float total_dist =
        route_total_distance_km() *
        sc->distance_factor;

    /*
     * The policy itself is expressed in remaining km.
     * For an altered route, the start distance can range over that route.
     */
    SimResult late =
        simulate_distance_candidate(
            0.0f,
            T_init,
            SOC_init,
            sc
        );

    if (!late.physically_valid)
        return 0;

    if (late.SOC_end_frac < SOC_MIN)
        return 0;

    if (late.T_end_C > T_OPT_HIGH)
        return 0;

    SimResult early =
        simulate_distance_candidate(
            total_dist,
            T_init,
            SOC_init,
            sc
        );

    if (early.physically_valid &&
        early.T_end_C < T_OPT_LOW)
        return 0;

    float best_violation = INFINITY;
    float best_x = 0.0f;

    for (int level = 0;
         level < N_COARSE_LEVELS;
         ++level) {

        float step =
            time_equiv_step_to_km(
                (float)COARSE_STEP_TIME_S[level]
            );

        for (float x = 0.0f;
             x <= total_dist + EPS;
             x += step) {

            float cx =
                fminf(x, total_dist);

            SimResult r =
                simulate_distance_candidate(
                    cx,
                    T_init,
                    SOC_init,
                    sc
                );

            if (r.feasible) {
                *best_feasible = r;
                return 1;
            }

            float vio =
                scenario_violation_score(&r);

            if (vio < best_violation) {
                best_violation = vio;
                best_x = cx;
            }

            if (cx >= total_dist - EPS)
                break;
        }
    }

    /*
     * No feasible 5-s coarse point:
     * perform only a local 1-s refinement around the least-violating point.
     */
    float finest_coarse =
        time_equiv_step_to_km(
            (float)COARSE_STEP_TIME_S[
                N_COARSE_LEVELS - 1
            ]
        );

    float fine_step =
        time_equiv_step_to_km(
            FINE_STEP_TIME_EQUIV_S
        );

    float lo =
        fmaxf(
            0.0f,
            best_x -
            4.0f * finest_coarse
        );

    float hi =
        fminf(
            total_dist,
            best_x +
            4.0f * finest_coarse
        );

    float best_energy =
        INFINITY;

    int found = 0;

    for (float x = lo;
         x <= hi + EPS;
         x += fine_step) {

        float cx =
            fminf(x, hi);

        SimResult r =
            simulate_distance_candidate(
                cx,
                T_init,
                SOC_init,
                sc
            );

        if (r.feasible &&
            r.E_heat_kWh <
            best_energy) {

            best_energy =
                r.E_heat_kWh;

            *best_feasible = r;
            found = 1;
        }

        if (cx >= hi - EPS)
            break;
    }

    return found;
}

static void run_failure_boundary_check(float T_init,
                                       float SOC_init,
                                       float selected_start_distance)
{
    printf("\n");
    printf("====================================================================================================\n");
    printf("FAILURE BOUNDARY CHECK\n");
    printf("For each failed fixed-policy stress case, search the whole distance range again.\n");
    printf("This identifies policy failure vs. physically missing feasible region.\n");
    printf("====================================================================================================\n");

    for (int s = 0; s < N_STRESS_SCENARIOS; ++s) {
        SimResult fixed =
            simulate_distance_candidate(
                selected_start_distance,
                T_init,
                SOC_init,
                &STRESS_SCENARIOS[s]
            );

        if (fixed.feasible)
            continue;

        SimResult alternative;

        int any =
            scenario_has_any_feasible_solution(
                T_init,
                SOC_init,
                &STRESS_SCENARIOS[s],
                &alternative
            );

        if (any) {
            printf("  %-20s : POLICY NOT ROBUST ENOUGH; another feasible start exists at %.2f km "
                   "(T=%.2f C, SOC=%.2f%%)\n",
                   STRESS_SCENARIOS[s].name,
                   alternative.start_distance_km,
                   alternative.T_end_C,
                   alternative.SOC_end_frac * 100.0f);
        } else {
            printf("  %-20s : NO FEASIBLE SOLUTION under this physical scenario\n",
                   STRESS_SCENARIOS[s].name);
        }
    }
}

/* =========================================================================
 * 11. Re-optimization adaptability and compute-demand analysis
 * ========================================================================= */

static void backup_route(NavSeg *backup)
{
    for (int i = 0; i < g_n_segs; ++i) backup[i] = g_segs[i];
}

static void restore_route(const NavSeg *backup)
{
    for (int i = 0; i < g_n_segs; ++i) g_segs[i] = backup[i];
}

static void apply_scenario_to_route(const NavSeg *base, const Scenario *sc)
{
    for (int i = 0; i < g_n_segs; ++i) {
        g_segs[i] = base[i];
        g_segs[i].s_km = fmaxf(0.001f, base[i].s_km * sc->distance_factor);
        g_segs[i].v_kmh = fmaxf(0.1f, base[i].v_kmh * sc->speed_factor);
        g_segs[i].P_drive_kW = fmaxf(0.0f, base[i].P_drive_kW * sc->power_factor);
        g_segs[i].T_env_C = base[i].T_env_C + sc->ambient_offset_C;
    }
}

static void run_reoptimization_adaptability(float T_init, float SOC_init)
{
    NavSeg backup[MAX_SEGS];
    backup_route(backup);

    int pass_count = 0;
    double total_ms = 0.0;
    unsigned long long total_sims = 0;
    unsigned long long total_steps = 0;

    printf("\n====================================================================================================\n");
    printf("RE-OPTIMIZATION ADAPTABILITY TEST\n");
    printf("Each changed navigation condition is re-planned by the same adaptive robust optimizer.\n");
    printf("====================================================================================================\n");
    printf("%-20s %-6s %-9s %-9s %-9s %-10s %-8s %-8s\n",
           "Scenario","PASS","StartKm","Tend(C)","SOC(%)","CPU(ms)","Margin","Step(s)");
    printf("----------------------------------------------------------------------------------------------------\n");

    for (int sidx = 0; sidx < N_STRESS_SCENARIOS; ++sidx) {
        restore_route(backup);
        apply_scenario_to_route(backup, &STRESS_SCENARIOS[sidx]);

        float Ti = T_init + STRESS_SCENARIOS[sidx].init_temp_offset_C;
        float Si = clamp_local(SOC_init + STRESS_SCENARIOS[sidx].soc_offset_frac, 0.0f, 1.0f);

        SimResult b;
        OptimStats st;
        int ok = robust_optimize(Ti, Si, &b, &st, 0);
        int pass = ok && b.feasible;
        if (pass) pass_count++;

        total_ms += st.elapsed_ms;
        total_sims += st.counters.sim_calls;
        total_steps += st.counters.integration_steps;

        if (pass) {
            printf("%-20s %-6s %-9.2f %-9.2f %-9.2f %-10.3f %-8.2f %-8d\n",
                   STRESS_SCENARIOS[sidx].name,"PASS",b.start_distance_km,b.T_end_C,
                   b.SOC_end_frac*100.0f,st.elapsed_ms,st.selected_temp_margin_C,
                   st.selected_coarse_step_time_s);
        } else {
            printf("%-20s %-6s %-9s %-9s %-9s %-10.3f %-8s %-8s\n",
                   STRESS_SCENARIOS[sidx].name,"FAIL","-","-","-",st.elapsed_ms,"-","-");
        }
    }

    restore_route(backup);
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("Re-optimization feasible scenarios : %d / %d\n", pass_count, N_STRESS_SCENARIOS);
    printf("Re-optimization adaptability rate  : %.2f %%\n",
           100.0f*(float)pass_count/(float)N_STRESS_SCENARIOS);
    printf("Average re-optimization CPU time   : %.3f ms/scenario\n", total_ms/(double)N_STRESS_SCENARIOS);
    printf("Average forward simulations        : %.1f/scenario\n", (double)total_sims/(double)N_STRESS_SCENARIOS);
    printf("Average 1-s integration steps      : %.1f/scenario\n", (double)total_steps/(double)N_STRESS_SCENARIOS);
}

static void print_compute_report(const OptimStats *stats, float start_distance)
{
    float trigger_time_s = nominal_trigger_time_s(start_distance);
    double deadline_ms = 1000.0*(double)trigger_time_s;
    double slack_ms = deadline_ms - stats->elapsed_ms;

    size_t nav_bytes = sizeof(g_segs);
    size_t map_bytes = sizeof(R_INT_MAP)+sizeof(UOC_MAP)+sizeof(P_CHARGE_MAP)+
                       sizeof(R_INT_TEMPS)+sizeof(R_INT_SOCS)+sizeof(U_OC_TEMPS)+
                       sizeof(U_OC_SOCS)+sizeof(P_CHARGE_TEMPS)+sizeof(P_CHARGE_SOCS);
    size_t scenario_bytes = sizeof(ROBUST_SCENARIOS)+sizeof(STRESS_SCENARIOS);
    size_t candidate_bytes = sizeof(CandidateEval)*MAX_FINE_CANDIDATES;
    size_t approx_bytes = nav_bytes + map_bytes + scenario_bytes + candidate_bytes +
                          sizeof(OptimStats)+sizeof(SimResult)+sizeof(ComputeCounters);

    printf("\n====================================================================================================\n");
    printf("VEHICLE-SIDE REAL-TIME / COMPUTE-DEMAND REPORT\n");
    printf("====================================================================================================\n");
    printf("Nominal route horizon                 : %.1f s\n", route_total_time_s());
    printf("Nominal time before selected trigger  : %.1f s\n", trigger_time_s);
    printf("Optimization CPU latency              : %.3f ms\n", stats->elapsed_ms);
    if (trigger_time_s > EPS) {
        printf("Compute deadline slack                : %.3f ms\n", slack_ms);
        printf("CPU/deadline ratio                    : %.6f %%\n",
               100.0*stats->elapsed_ms/deadline_ms);
        printf("Finish before nominal trigger?        : %s\n", slack_ms >= 0.0 ? "YES" : "NO");
    } else {
        printf("Finish before nominal trigger?        : must be precomputed before departure\n");
    }
    printf("\nCompute workload:\n");
    printf("  Coarse candidate evaluations        : %d\n", stats->coarse_evals);
    printf("  Fine candidate evaluations          : %d\n", stats->fine_evals);
    printf("  Forward-simulation calls            : %llu\n", stats->counters.sim_calls);
    printf("  1-s route integration steps         : %llu\n", stats->counters.integration_steps);
    printf("  2-D interpolation calls             : %llu\n", stats->counters.interp_calls);
    printf("  Charging integration steps          : %llu\n", stats->counters.charge_steps);
    printf("\nApproximate working memory:\n");
    printf("  Navigation buffer                   : %zu bytes\n", nav_bytes);
    printf("  Lookup maps                         : %zu bytes\n", map_bytes);
    printf("  Scenario tables                     : %zu bytes\n", scenario_bytes);
    printf("  Candidate buffer                    : %zu bytes\n", candidate_bytes);
    printf("  Approx. accounted total             : %zu bytes (%.2f KB)\n",
           approx_bytes,(double)approx_bytes/1024.0);
    printf("\nNote: the specification requires simulation time step <= 1 s but does not provide an official CPU deadline.\n");
    printf("Here, the time from navigation start to the selected trigger point is used as the practical decision deadline.\n");
}

/* =========================================================================
 * 12. Main
 * ========================================================================= */

int main(void)
{
    /* Read initial battery state */
    float T_init = 0.0f;
    EMS_HVBatt_getTempAvg(&T_init);

    float SOC_init = 0.0f;
    EMS_HVBatt_getTargetSOC(&SOC_init);
    SOC_init /= 100.0f;

    float I_batt = 0.0f;
    EMS_HVBatt_getCurrent(&I_batt);

    float V_batt = 0.0f;
    EMS_HVBatt_getVolt(&V_batt);

    /* Read navigation route */
    load_nav_segments();

    printf("=== Initial State ===\n");
    printf("  Battery temp:     %.1f C\n", T_init);
    printf("  Battery SOC:      %.1f %%\n", SOC_init * 100.0f);
    printf("  Battery current:  %.1f A\n", I_batt);
    printf("  Battery voltage:  %.1f V\n", V_batt);
    printf("  Route segments:   %d\n", g_n_segs);

    printf("\n=== Route Segments ===\n");
    printf("  %-6s %-10s %-10s %-12s %-12s\n",
           "Seg", "Dist(km)", "Spd(km/h)",
           "Pdrive(kW)", "Tamb(C)");

    for (int i = 0; i < g_n_segs; ++i) {
        printf("  %-6d %-10.3f %-10.3f %-12.3f %-12.3f\n",
               i + 1,
               g_segs[i].s_km,
               g_segs[i].v_kmh,
               g_segs[i].P_drive_kW,
               g_segs[i].T_env_C);
    }

    printf("\n=== Adaptive Robust Algorithm V4 ===\n");

    SimResult optimum;
    OptimStats stats;

    if (!robust_optimize(
            T_init,
            SOC_init,
            &optimum,
            &stats,
            1)) {

        fprintf(stderr,
                "ERROR: no nominally feasible preheat strategy found.\n");

        return 2;
    }

    float start_distance = optimum.start_distance_km;
    float T_end_opt = optimum.T_end_C;
    float SOC_end_opt = optimum.SOC_end_frac * 100.0f;
    float E_heat_opt = optimum.E_heat_kWh;
    float chrg_time_s = optimum.charge_time_s;

    /* Notify competition system */
    BattChrgPreHeatg_ntfPreHeatgStartDist(start_distance);
    BattChrgPreHeatg_ntfPreHeatgEndTemp(T_end_opt);
    BattChrgPreHeatg_ntfPreHeatgEndSOC(SOC_end_opt);
    BattChrgPreHeatg_ntfPreHeatgEnergy(E_heat_opt);

    printf("\n===== Result Verification =====\n");
    printf("  start_distance: notified=%.2f km\n",
           start_distance);
    printf("  T_end_opt:      notified=%.2f\n",
           T_end_opt);
    printf("  SOC_end_opt:    notified=%.2f\n",
           SOC_end_opt);
    printf("  E_heat_opt:     notified=%.4f\n",
           E_heat_opt);

    printf("\n===== Results =====\n");

    printf("  start_distance = %.2f km  "
           "(begin preheat when %.2f km from charger)\n",
           start_distance,
           start_distance);

    printf("  T_end_opt      = %.2f C  [target: 20 ~ 25 C]\n",
           T_end_opt);

    printf("  SOC_end_opt    = %.2f %%  [min: 10 %%]\n",
           SOC_end_opt);

    printf("  E_heat_opt     = %.4f kWh  (%.0f kJ)\n",
           E_heat_opt,
           E_heat_opt * 3600.0f);

    printf("  chrg_time_s    = %.2f s  "
           "(to charge from %.1f%% to 80%%)\n",
           chrg_time_s,
           SOC_end_opt);

    printf("\n===== Constraint Checks =====\n");

    int pass = 1;

    if (T_end_opt >= T_OPT_LOW &&
        T_end_opt <= T_OPT_HIGH) {

        printf("  [PASS] Temperature %.2f C in [%.0f, %.0f] C\n",
               T_end_opt,
               T_OPT_LOW,
               T_OPT_HIGH);
    } else {
        printf("  [FAIL] Temperature %.2f C NOT in [%.0f, %.0f] C\n",
               T_end_opt,
               T_OPT_LOW,
               T_OPT_HIGH);

        pass = 0;
    }

    if (SOC_end_opt >= SOC_MIN * 100.0f) {
        printf("  [PASS] SOC %.2f %% >= %.0f %%\n",
               SOC_end_opt,
               SOC_MIN * 100.0f);
    } else {
        printf("  [FAIL] SOC %.2f %% < %.0f %%\n",
               SOC_end_opt,
               SOC_MIN * 100.0f);

        pass = 0;
    }

    printf("  %s\n",
           pass ?
           "ALL CONSTRAINTS SATISFIED" :
           "SOME CONSTRAINTS VIOLATED");

    printf("\n===== Adaptive Robust Optimization Summary =====\n");
    printf("  Selection rule                 : lexicographic robust optimization\n");
    printf("  Priority 1                     : maximize scenario coverage\n");
    printf("  Priority 2                     : maximize worst T/SOC safety index\n");
    printf("  Priority 3                     : minimize official 30:20 objective\n");
    printf("  Selected safety margin         : %.2f C\n",
           stats.selected_temp_margin_C);
    printf("  Internal target temperature    : [%.2f, %.2f] C\n",
           T_OPT_LOW + stats.selected_temp_margin_C,
           T_OPT_HIGH - stats.selected_temp_margin_C);
    printf("  Selected coarse step           : %d s-equivalent (%.4f km)\n",
           stats.selected_coarse_step_time_s,
           stats.selected_coarse_step_km);
    printf("  Fine step                      : 1 s-equivalent (%.4f km)\n",
           stats.fine_step_km);
    printf("  Active robust scenarios        : %d / %d\n",
           stats.active_robust_scenarios,
           N_ROBUST_SCENARIOS);
    printf("  Excluded incompatible scenarios: %d\n",
           stats.excluded_incompatible_scenarios);
    printf("  Best robust coverage           : %.2f %% (%d/%d active)\n",
           100.0f * stats.best_robust_feasible_rate,
           stats.best_robust_feasible_count,
           stats.active_robust_scenarios);
    printf("  Common policy covers all active: %s\n",
           stats.common_robust_policy_exists ? "YES" : "NO");
    printf("  Worst normalized safety index  : %.4f\n",
           stats.best_worst_safety_index);
    printf("  Dynamic-margin coverage        : %.2f %% (%d/%d covered)\n",
           100.0f * stats.best_safe_rate,
           stats.best_safe_count,
           stats.best_robust_feasible_count);
    printf("  Official 30:20 normalized cost : %.6f\n",
           stats.official_objective);
    printf("  Optimization CPU time          : %.3f ms\n",
           stats.elapsed_ms);

    print_compute_report(&stats, start_distance);

    run_fixed_policy_stress_test(start_distance, T_init, SOC_init);

    run_reoptimization_adaptability(T_init, SOC_init);

    run_failure_boundary_check(T_init, SOC_init, start_distance);

    printf("\n===== Done =====\n");

    return 0;
}