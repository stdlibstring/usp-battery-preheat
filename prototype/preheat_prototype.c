#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MAKE_DIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MAKE_DIR(path) mkdir(path, 0777)
#endif

#define MAX_SEGS 16
#ifndef SIM_DT_S
#define SIM_DT_S 0.5
#endif
#ifndef CANDIDATE_STEP_S
#define CANDIDATE_STEP_S 1.0
#endif

#define M_BAT_KG 400.0
#define CP_J_PER_KG_C 1000.0
#define C_NOM_AH 100.0
#define U_NOM_V 350.0
#define P_HEAT_MAX_W 6000.0
#define ETA_HEAT 0.95
#define H0_W_PER_C 50.0
#define KV_W_PER_C_KMH 0.5
#define T_OPT_LOW_C 20.0
#define T_OPT_HIGH_C 25.0
#define SOC_MIN_PCT 10.0
#define SOC_TARGET_PCT 80.0

static const double MAP_SOC_PCT[8] = {0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0, 100.0};
static const double ELEC_TEMP_C[7] = {-20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0};
static const double CHARGE_TEMP_C[7] = {-20.0, -10.0, 0.0, 10.0, 23.0, 30.0, 40.0};

static const double R_INT_OHM[7][8] = {
    {0.150, 0.120, 0.100, 0.090, 0.085, 0.082, 0.080, 0.078},
    {0.100, 0.085, 0.075, 0.070, 0.065, 0.062, 0.060, 0.058},
    {0.070, 0.060, 0.052, 0.048, 0.045, 0.043, 0.042, 0.041},
    {0.050, 0.042, 0.038, 0.035, 0.033, 0.032, 0.031, 0.030},
    {0.038, 0.032, 0.029, 0.027, 0.025, 0.024, 0.023, 0.023},
    {0.032, 0.028, 0.025, 0.023, 0.022, 0.021, 0.020, 0.020},
    {0.028, 0.025, 0.022, 0.021, 0.020, 0.019, 0.019, 0.018},
};

static const double U_OC_V[7][8] = {
    {295, 320, 340, 352, 365, 375, 382, 410},
    {298, 323, 343, 355, 368, 378, 385, 413},
    {300, 325, 345, 357, 370, 380, 387, 415},
    {302, 327, 347, 359, 372, 382, 389, 417},
    {303, 328, 348, 360, 373, 383, 390, 418},
    {304, 329, 349, 361, 374, 384, 391, 419},
    {305, 330, 350, 362, 375, 385, 392, 420},
};

static const double P_CHARGE_KW[7][8] = {
    {10, 12, 15, 18, 20, 18, 15, 10},
    {20, 25, 30, 35, 38, 35, 28, 20},
    {40, 50, 60, 65, 68, 65, 55, 40},
    {60, 70, 80, 85, 88, 85, 70, 50},
    {75, 85, 90, 92, 90, 88, 75, 55},
    {70, 80, 88, 90, 88, 85, 70, 50},
    {50, 60, 70, 72, 70, 65, 55, 40},
};

typedef struct {
    double s_km;
    double v_kmh;
    double p_drive_kw;
    double t_env_c;
} NavSeg;

typedef struct {
    NavSeg segs[MAX_SEGS];
    int count;
    double total_distance_km;
    double total_time_s;
} Route;

typedef struct {
    bool valid;
    bool feasible;
    double start_time_s;
    double start_distance_km;
    double end_temp_c;
    double end_soc_pct;
    double heat_energy_kwh;
    double charge_time_s;
    double charge_power_kw;
} SimResult;

typedef struct {
    size_t count;
    size_t capacity;
    double *time_s;
    double *temp_c;
    double *soc_pct;
    double *current_a;
    double *heating_kw;
    double *remaining_km;
} Trace;

typedef struct {
    SimResult sim;
    double score;
} Candidate;

static double clamp_double(double x, double low, double high)
{
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

static int lower_index(const double *axis, int n, double x)
{
    if (x <= axis[0]) return 0;
    if (x >= axis[n - 1]) return n - 2;
    for (int i = 0; i < n - 1; ++i) {
        if (x <= axis[i + 1]) return i;
    }
    return n - 2;
}

static double interp2(const double temp_axis[7], const double soc_axis[8],
                      const double map[7][8], double temp_c, double soc_pct)
{
    const double t = clamp_double(temp_c, temp_axis[0], temp_axis[6]);
    const double s = clamp_double(soc_pct, soc_axis[0], soc_axis[7]);
    const int ti = lower_index(temp_axis, 7, t);
    const int si = lower_index(soc_axis, 8, s);
    const double ta = (t - temp_axis[ti]) / (temp_axis[ti + 1] - temp_axis[ti]);
    const double sa = (s - soc_axis[si]) / (soc_axis[si + 1] - soc_axis[si]);
    const double v0 = map[ti][si] + sa * (map[ti][si + 1] - map[ti][si]);
    const double v1 = map[ti + 1][si] + sa * (map[ti + 1][si + 1] - map[ti + 1][si]);
    return v0 + ta * (v1 - v0);
}

static double resistance_ohm(double temp_c, double soc_pct)
{
    return interp2(ELEC_TEMP_C, MAP_SOC_PCT, R_INT_OHM, temp_c, soc_pct);
}

static double open_circuit_voltage_v(double temp_c, double soc_pct)
{
    return interp2(ELEC_TEMP_C, MAP_SOC_PCT, U_OC_V, temp_c, soc_pct);
}

static double charge_power_kw(double temp_c, double soc_pct)
{
    return interp2(CHARGE_TEMP_C, MAP_SOC_PCT, P_CHARGE_KW, temp_c, soc_pct);
}

static bool discharge_current_a(double power_w, double uoc_v, double r_ohm, double *current_a)
{
    const double disc = uoc_v * uoc_v - 4.0 * r_ohm * power_w;
    if (disc < 0.0 || r_ohm <= 0.0) return false;
    const double denom = uoc_v + sqrt(disc);
    if (fabs(denom) < 1e-12) return false;
    *current_a = 2.0 * power_w / denom;
    return isfinite(*current_a);
}

static bool trace_init(Trace *trace, size_t capacity)
{
    memset(trace, 0, sizeof(*trace));
    trace->capacity = capacity;
    trace->time_s = calloc(capacity, sizeof(double));
    trace->temp_c = calloc(capacity, sizeof(double));
    trace->soc_pct = calloc(capacity, sizeof(double));
    trace->current_a = calloc(capacity, sizeof(double));
    trace->heating_kw = calloc(capacity, sizeof(double));
    trace->remaining_km = calloc(capacity, sizeof(double));
    return trace->time_s && trace->temp_c && trace->soc_pct && trace->current_a &&
           trace->heating_kw && trace->remaining_km;
}

static void trace_free(Trace *trace)
{
    free(trace->time_s);
    free(trace->temp_c);
    free(trace->soc_pct);
    free(trace->current_a);
    free(trace->heating_kw);
    free(trace->remaining_km);
    memset(trace, 0, sizeof(*trace));
}

static bool trace_push(Trace *trace, double time_s, double temp_c, double soc_pct,
                       double current_a, double heating_kw, double remaining_km)
{
    if (!trace) return true;
    if (trace->count >= trace->capacity) return false;
    const size_t i = trace->count++;
    trace->time_s[i] = time_s;
    trace->temp_c[i] = temp_c;
    trace->soc_pct[i] = soc_pct;
    trace->current_a[i] = current_a;
    trace->heating_kw[i] = heating_kw;
    trace->remaining_km[i] = remaining_km;
    return true;
}

static bool parse_route_csv(const char *path, Route *route)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open route CSV '%s': %s\n", path, strerror(errno));
        return false;
    }

    memset(route, 0, sizeof(*route));
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        NavSeg seg;
        if (sscanf(line, " %lf , %lf , %lf , %lf", &seg.s_km, &seg.v_kmh,
                   &seg.p_drive_kw, &seg.t_env_c) != 4) {
            continue; /* Header or comment. */
        }
        if (route->count >= MAX_SEGS) {
            fprintf(stderr, "Route has more than %d segments.\n", MAX_SEGS);
            fclose(fp);
            return false;
        }
        if (seg.s_km <= 0.0 || seg.v_kmh <= 0.0) {
            fprintf(stderr, "Route segment distance and speed must be positive.\n");
            fclose(fp);
            return false;
        }
        route->segs[route->count++] = seg;
        route->total_distance_km += seg.s_km;
        route->total_time_s += seg.s_km / seg.v_kmh * 3600.0;
    }
    fclose(fp);
    if (route->count == 0) {
        fprintf(stderr, "No valid route segments found in '%s'.\n", path);
        return false;
    }
    return true;
}

static double remaining_distance_at_time(const Route *route, double time_s)
{
    double remaining = route->total_distance_km;
    double elapsed = 0.0;
    for (int i = 0; i < route->count; ++i) {
        const double seg_time = route->segs[i].s_km / route->segs[i].v_kmh * 3600.0;
        if (time_s >= elapsed + seg_time) {
            remaining -= route->segs[i].s_km;
            elapsed += seg_time;
        } else {
            const double local_time = clamp_double(time_s - elapsed, 0.0, seg_time);
            remaining -= route->segs[i].v_kmh * local_time / 3600.0;
            break;
        }
    }
    return fmax(0.0, remaining);
}

static SimResult simulate_route(const Route *route, double initial_temp_c,
                                double initial_soc_pct, double start_time_s, Trace *trace)
{
    SimResult out;
    memset(&out, 0, sizeof(out));
    out.start_time_s = clamp_double(start_time_s, 0.0, route->total_time_s);
    out.start_distance_km = remaining_distance_at_time(route, out.start_time_s);

    double temp_c = initial_temp_c;
    double soc_pct = initial_soc_pct;
    double heat_energy_kwh = 0.0;
    double time_s = 0.0;
    double remaining_km = route->total_distance_km;
    double last_current_a = 0.0;
    double last_heating_kw = 0.0;

    for (int seg_i = 0; seg_i < route->count; ++seg_i) {
        const NavSeg *seg = &route->segs[seg_i];
        double seg_remaining_km = seg->s_km;

        while (seg_remaining_km > 1e-10) {
            bool heating_on = time_s + 1e-9 >= out.start_time_s &&
                              out.start_time_s < route->total_time_s - 1e-9;
            double dt_s = fmin(SIM_DT_S, seg_remaining_km / seg->v_kmh * 3600.0);
            if (!heating_on && out.start_time_s > time_s + 1e-9 &&
                out.start_time_s < time_s + dt_s - 1e-9) {
                dt_s = out.start_time_s - time_s;
            }

            const double p_heat_w = heating_on ? P_HEAT_MAX_W : 0.0;
            const double r_ohm = resistance_ohm(temp_c, soc_pct);
            const double uoc_v = open_circuit_voltage_v(temp_c, soc_pct);
            const double p_bat_w = seg->p_drive_kw * 1000.0 + p_heat_w;
            double current_a = 0.0;
            if (!discharge_current_a(p_bat_w, uoc_v, r_ohm, &current_a)) {
                return out;
            }

            if (!trace_push(trace, time_s, temp_c, soc_pct, current_a,
                            p_heat_w / 1000.0, remaining_km)) {
                return out;
            }

            const double q_gen_w = current_a * current_a * r_ohm;
            const double q_heat_w = ETA_HEAT * p_heat_w;
            const double k_env_w_per_c = H0_W_PER_C + KV_W_PER_C_KMH * seg->v_kmh;
            const double q_loss_w = k_env_w_per_c * (temp_c - seg->t_env_c);
            temp_c += (q_gen_w + q_heat_w - q_loss_w) /
                      (M_BAT_KG * CP_J_PER_KG_C) * dt_s;
            soc_pct -= current_a * dt_s / (C_NOM_AH * 3600.0) * 100.0;
            heat_energy_kwh += p_heat_w * dt_s / 3.6e6;

            const double traveled_km = seg->v_kmh * dt_s / 3600.0;
            seg_remaining_km = fmax(0.0, seg_remaining_km - traveled_km);
            remaining_km = fmax(0.0, remaining_km - traveled_km);
            time_s += dt_s;
            last_current_a = current_a;
            last_heating_kw = p_heat_w / 1000.0;

            if (!isfinite(temp_c) || !isfinite(soc_pct) || soc_pct < -1.0) {
                return out;
            }
        }
    }

    if (!trace_push(trace, route->total_time_s, temp_c, soc_pct, last_current_a,
                    last_heating_kw, 0.0)) {
        return out;
    }

    out.valid = true;
    out.end_temp_c = temp_c;
    out.end_soc_pct = soc_pct;
    out.heat_energy_kwh = heat_energy_kwh;
    out.charge_power_kw = charge_power_kw(temp_c, soc_pct);
    const double energy_to_target_kwh = fmax(0.0, (SOC_TARGET_PCT - soc_pct) / 100.0 *
                                                      C_NOM_AH * U_NOM_V / 1000.0);
    out.charge_time_s = out.charge_power_kw > 0.0
                            ? energy_to_target_kwh / out.charge_power_kw * 3600.0
                            : DBL_MAX;
    out.feasible = temp_c >= T_OPT_LOW_C && temp_c <= T_OPT_HIGH_C &&
                   soc_pct >= SOC_MIN_PCT;
    return out;
}

static bool optimize(const Route *route, double initial_temp_c, double initial_soc_pct,
                     Candidate *best, int *feasible_count, const char *output_dir)
{
    const int max_candidates = (int)ceil(route->total_time_s / CANDIDATE_STEP_S) + 2;
    Candidate *feasible = calloc((size_t)max_candidates, sizeof(Candidate));
    if (!feasible) return false;

    int count = 0;
    for (double start_s = 0.0; start_s <= route->total_time_s + 1e-9;
         start_s += CANDIDATE_STEP_S) {
        if (start_s > route->total_time_s) start_s = route->total_time_s;
        SimResult sim = simulate_route(route, initial_temp_c, initial_soc_pct, start_s, NULL);
        if (sim.valid && sim.feasible) {
            feasible[count++].sim = sim;
        }
        if (start_s >= route->total_time_s) break;
    }

    *feasible_count = count;
    if (count == 0) {
        free(feasible);
        return false;
    }

    double t_min = DBL_MAX, t_max = -DBL_MAX;
    double e_min = DBL_MAX, e_max = -DBL_MAX;
    for (int i = 0; i < count; ++i) {
        t_min = fmin(t_min, feasible[i].sim.charge_time_s);
        t_max = fmax(t_max, feasible[i].sim.charge_time_s);
        e_min = fmin(e_min, feasible[i].sim.heat_energy_kwh);
        e_max = fmax(e_max, feasible[i].sim.heat_energy_kwh);
    }

    bool have_best = false;
    for (int i = 0; i < count; ++i) {
        const double time_score = (t_max - t_min) > 1e-12
                                      ? 30.0 * (t_max - feasible[i].sim.charge_time_s) /
                                            (t_max - t_min)
                                      : 30.0;
        const double energy_score = (e_max - e_min) > 1e-12
                                        ? 20.0 * (e_max - feasible[i].sim.heat_energy_kwh) /
                                              (e_max - e_min)
                                        : 20.0;
        feasible[i].score = time_score + energy_score;
        if (!have_best || feasible[i].score > best->score + 1e-12 ||
            (fabs(feasible[i].score - best->score) <= 1e-12 &&
             feasible[i].sim.heat_energy_kwh < best->sim.heat_energy_kwh)) {
            *best = feasible[i];
            have_best = true;
        }
    }

    if (output_dir) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/candidate_sweep.csv", output_dir);
        FILE *fp = fopen(path, "w");
        if (!fp) {
            free(feasible);
            return false;
        }
        fprintf(fp, "start_time_s,start_distance_km,end_temp_c,end_soc_pct,heat_energy_kwh,charge_time_s,score_50\n");
        for (int i = 0; i < count; ++i) {
            fprintf(fp, "%.6f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    feasible[i].sim.start_time_s,
                    feasible[i].sim.start_distance_km,
                    feasible[i].sim.end_temp_c,
                    feasible[i].sim.end_soc_pct,
                    feasible[i].sim.heat_energy_kwh,
                    feasible[i].sim.charge_time_s,
                    feasible[i].score);
        }
        fclose(fp);
    }

    free(feasible);
    return have_best;
}

static bool ensure_directory(const char *path)
{
    if (MAKE_DIR(path) == 0) return true;
    return errno == EEXIST;
}

static bool write_trace_csv(const char *output_dir, const Trace *trace)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/trajectory.csv", output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    fprintf(fp, "time_s,temp_c,soc_pct,current_a,heating_kw,remaining_km\n");
    for (size_t i = 0; i < trace->count; ++i) {
        fprintf(fp, "%.6f,%.9f,%.9f,%.9f,%.6f,%.9f\n", trace->time_s[i],
                trace->temp_c[i], trace->soc_pct[i], trace->current_a[i],
                trace->heating_kw[i], trace->remaining_km[i]);
    }
    fclose(fp);
    return true;
}

static bool write_summary_csv(const char *output_dir, const Candidate *best,
                              int feasible_count, double initial_temp_c,
                              double initial_soc_pct, const Route *route)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/summary.csv", output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    fprintf(fp, "metric,value\n");
    fprintf(fp, "initial_temp_c,%.9f\n", initial_temp_c);
    fprintf(fp, "initial_soc_pct,%.9f\n", initial_soc_pct);
    fprintf(fp, "route_segments,%d\n", route->count);
    fprintf(fp, "route_distance_km,%.9f\n", route->total_distance_km);
    fprintf(fp, "route_time_s,%.9f\n", route->total_time_s);
    fprintf(fp, "feasible_candidates,%d\n", feasible_count);
    fprintf(fp, "score_50,%.9f\n", best->score);
    fprintf(fp, "start_time_s,%.9f\n", best->sim.start_time_s);
    fprintf(fp, "start_distance_km,%.9f\n", best->sim.start_distance_km);
    fprintf(fp, "end_temp_c,%.9f\n", best->sim.end_temp_c);
    fprintf(fp, "end_soc_pct,%.9f\n", best->sim.end_soc_pct);
    fprintf(fp, "heat_energy_kwh,%.9f\n", best->sim.heat_energy_kwh);
    fprintf(fp, "charge_power_kw,%.9f\n", best->sim.charge_power_kw);
    fprintf(fp, "charge_time_s,%.9f\n", best->sim.charge_time_s);
    fclose(fp);
    return true;
}

static int self_test(void)
{
    const double eps = 1e-9;
    if (fabs(resistance_ohm(20.0, 50.0) - 0.025) > eps) return 1;
    if (fabs(open_circuit_voltage_v(-10.0, 70.0) - 378.0) > eps) return 2;
    if (fabs(charge_power_kw(23.0, 30.0) - 92.0) > eps) return 3;
    const double midpoint = resistance_ohm(15.0, 40.0);
    if (!(midpoint > 0.025 && midpoint < 0.035)) return 4;
    double current = 0.0;
    if (!discharge_current_a(20000.0, 370.0, 0.03, &current) || current <= 0.0) return 5;
    puts("SELF_TEST_PASS");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) return self_test();
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "Usage: %s ROUTE.csv OUTPUT_DIR [T_INIT_C] [SOC_INIT_PCT]\n", argv[0]);
        return 2;
    }

    const double initial_temp_c = argc >= 4 ? strtod(argv[3], NULL) : 0.0;
    const double initial_soc_pct = argc >= 5 ? strtod(argv[4], NULL) : 65.0;
    Route route;
    if (!parse_route_csv(argv[1], &route)) return 3;
    if (!ensure_directory(argv[2])) {
        fprintf(stderr, "Cannot create output directory '%s'.\n", argv[2]);
        return 4;
    }

    Candidate best;
    memset(&best, 0, sizeof(best));
    int feasible_count = 0;
    if (!optimize(&route, initial_temp_c, initial_soc_pct, &best, &feasible_count,
                  argv[2])) {
        fprintf(stderr, "No feasible preheat start found for this route and initial state.\n");
        return 5;
    }

    Trace trace;
    const size_t capacity = (size_t)ceil(route.total_time_s / SIM_DT_S) +
                            (size_t)route.count + 8;
    if (!trace_init(&trace, capacity)) {
        fprintf(stderr, "Cannot allocate final trajectory trace.\n");
        return 6;
    }
    SimResult verified = simulate_route(&route, initial_temp_c, initial_soc_pct,
                                        best.sim.start_time_s, &trace);
    if (!verified.valid || !verified.feasible) {
        fprintf(stderr, "Best candidate failed verification.\n");
        trace_free(&trace);
        return 7;
    }
    best.sim = verified;

    if (!write_trace_csv(argv[2], &trace) ||
        !write_summary_csv(argv[2], &best, feasible_count, initial_temp_c,
                           initial_soc_pct, &route)) {
        fprintf(stderr, "Failed to write output files.\n");
        trace_free(&trace);
        return 8;
    }
    trace_free(&trace);

    printf("===== Prototype Results =====\n");
    printf("feasible_candidates = %d\n", feasible_count);
    printf("score_50 = %.4f\n", best.score);
    printf("start_distance = %.4f km\n", best.sim.start_distance_km);
    printf("T_end_opt = %.4f C\n", best.sim.end_temp_c);
    printf("SOC_end_opt = %.4f %%\n", best.sim.end_soc_pct);
    printf("E_heat_opt = %.6f kWh\n", best.sim.heat_energy_kwh);
    printf("chrg_time_s = %.4f s\n", best.sim.charge_time_s);
    printf("ALL CONSTRAINTS SATISFIED\n");
    return 0;
}
