/*
 * oracle_generator.c — High-accuracy oracle label generator
 *
 * Multi-level start-time search over the preheat physics kernel
 * (preheat_model.h, identical arithmetic to student_solution.c):
 *
 *   L0  10 s   global coarse sweep        -> keep ALL feasible intervals
 *   L1  1 s    sweep inside raw windows   (+ whole-route fallback if L0 empty)
 *   L2  0.1 s  around top-10 candidates and every interval endpoint
 *   L3  0.01 s audit only (--audit N)     -> verify L2 optimum stability
 *
 * Speed: one no-heating prefix simulation per case (shared by all
 * candidates), each candidate costs one partial step + heated suffix.
 * OpenMP parallel over cases, chunked checkpoint output, --start/--end
 * range and --resume for restartability.
 *
 * Two oracles per case:
 *   raw  : 20.00 <= T_end <= 25.00, SOC_end >= 10.00
 *   safe : 20.05 <= T_end <= 24.95, SOC_end >= 10.05
 *
 * Modes:
 *   --selftest                       reproduce official mock baseline
 *   --dt-study [N] [--seed S]        dt convergence study on N random cases
 *   --generate [--dt X] [--start A] [--end B] [--resume]
 *            [--out-prefix P] [--audit N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0777)
#endif

#include "preheat_model.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------ */
/* Search configuration                                                 */
/* ------------------------------------------------------------------ */
#define STEP_COARSE_S   10.0
#define STEP_MID_S       1.0
#define STEP_FINE_S      0.1
#define STEP_AUDIT_S     0.01
#define FINE_WINDOW_S    1.0
#define AUDIT_WINDOW_S   0.6
#define TOPK             10

#define RAW_T_LOW    20.00
#define RAW_T_HIGH   25.00
#define RAW_SOC_MIN  10.00
#define SAFE_T_LOW   20.05
#define SAFE_T_HIGH  24.95
#define SAFE_SOC_MIN 10.05

/* ------------------------------------------------------------------ */
/* Training data parsing (GB18030 bytes handled structurally: the      */
/* fields we need are pure ASCII; case headers are '=' lines with '|') */
/* ------------------------------------------------------------------ */
typedef struct {
    PhSeg segs[PH_MAX_SEGS];
    int   n_segs;
    long  id;            /* 1-based ordinal in the file */
} CaseData;

static CaseData *g_cases = NULL;
static long      g_n_cases = 0;

static char *read_whole_file(const char *path, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    const long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    buf[sz] = '\0';
    *len_out = (size_t)sz;
    return buf;
}

static long parse_dataset(const char *path)
{
    size_t len = 0;
    char *buf = read_whole_file(path, &len);
    if (!buf) {
        fprintf(stderr, "ERROR: cannot read %s\n", path);
        return -1;
    }

    long cap = 1024, n = 0;
    CaseData *cases = (CaseData *)malloc((size_t)cap * sizeof(CaseData));

    char *line = buf;
    int cur_seg = -1; /* -1 = not inside a case */
    while (line < buf + len) {
        char *eol = strchr(line, '\n');
        char *next = eol ? eol + 1 : buf + len;
        if (eol && eol > line && eol[-1] == '\r') eol[-1] = '\0';
        if (eol) eol[0] = '\0';

        const char *s = line;
        while (*s == ' ' || *s == '\t') ++s;

        if (s[0] == '=' && strchr(s, '|') != NULL) {
            /* close previous case, then open a new one at index n */
            if (cur_seg > 0) ++n;
            if (n >= cap) {
                cap *= 2;
                cases = (CaseData *)realloc(cases, (size_t)cap * sizeof(CaseData));
            }
            memset(&cases[n], 0, sizeof(CaseData));
            cases[n].id = n + 1;
            cur_seg = 0;
        } else if (cur_seg >= 0 && s[0] >= '0' && s[0] <= '9') {
            /* line = seg_id  s_km  v_kmh  P_drive_kW  T_env */
            double sid, a, b, c, d;
            if (sscanf(s, "%lf %lf %lf %lf %lf", &sid, &a, &b, &c, &d) == 5 &&
                cur_seg < PH_MAX_SEGS) {
                cases[n].segs[cur_seg].s_km = a;
                cases[n].segs[cur_seg].v_kmh = b;
                cases[n].segs[cur_seg].p_drive_kw = c;
                cases[n].segs[cur_seg].t_env_c = d;
                ++cur_seg;
                cases[n].n_segs = cur_seg;
            }
        }
        /* anything else (text headers) is ignored */

        line = next;
    }
    if (cur_seg > 0) ++n;

    free(buf);
    /* drop cases without any segment */
    long m = 0;
    for (long i = 0; i < n; ++i) {
        if (cases[i].n_segs > 0) cases[m++] = cases[i];
    }
    for (long i = 0; i < m; ++i) cases[i].id = i + 1;

    g_cases = cases;
    g_n_cases = m;
    return m;
}

/* ------------------------------------------------------------------ */
/* Feasible-candidate bookkeeping (thread-local, reused per case)      */
/* ------------------------------------------------------------------ */
typedef struct {
    double t_start;
    double dist_km;
    double end_temp;
    double end_soc;
    double heat_kwh;
    double charge_time;
    double charge_power;
    double score;      /* filled at finalize */
    double penalty;    /* only used for fallback tracking */
} Cand;

typedef struct {
    Cand  *v;
    int    n, cap;
    double tch_min, tch_max, e_min, e_max;
    int    have_fb;
    Cand   fb;              /* least-violation fallback */
    double fb_pen;
} CandSet;

/* thread-local: each OpenMP worker owns its candidate buffers */
static _Thread_local CandSet tls_raw, tls_safe;
static _Thread_local PhSnap *tls_snaps = NULL;
static _Thread_local int     tls_snap_cap = 0;
static _Thread_local double  tls_T_at_zero = -DBL_MAX; /* T_end(t_start=0) */

static void set_reset(CandSet *s)
{
    s->n = 0;
    s->tch_min = DBL_MAX;  s->tch_max = -DBL_MAX;
    s->e_min   = DBL_MAX;  s->e_max   = -DBL_MAX;
    s->have_fb = 0;
    s->fb_pen  = DBL_MAX;
}

static CandSet *set_grow(CandSet *s)
{
    if (s->v == NULL) {
        s->cap = 4096;
        s->v = (Cand *)malloc((size_t)s->cap * sizeof(Cand));
    } else if (s->n >= s->cap) {
        s->cap *= 2;
        s->v = (Cand *)realloc(s->v, (size_t)s->cap * sizeof(Cand));
    }
    if (s->v == NULL) {
        fprintf(stderr, "FATAL: out of memory\n");
        exit(2);
    }
    return s;
}

static void ensure_snaps(int needed)
{
    if (needed > tls_snap_cap) {
        tls_snap_cap = needed * 2 + 256;
        tls_snaps = (PhSnap *)realloc(tls_snaps,
                                      (size_t)tls_snap_cap * sizeof(PhSnap));
        if (tls_snaps == NULL) {
            fprintf(stderr, "FATAL: out of memory (snaps)\n");
            exit(2);
        }
    }
}

/* Evaluate one candidate; classifies into raw and/or safe sets. */
static void evaluate_candidate(const PhRoute *r, const PhPrefix *pf,
                               double dt, double t_start, long *n_eval)
{
    const PhSimResult sim = ph_simulate_candidate(r, pf, dt, t_start);
    ++(*n_eval);
    if (!sim.valid) return;
    if (t_start <= PH_NUM_EPS) tls_T_at_zero = sim.end_temp_c;

    Cand c;
    memset(&c, 0, sizeof(c));
    c.t_start      = t_start;
    c.dist_km      = ph_remaining_km_at_time(r, t_start);
    c.end_temp     = sim.end_temp_c;
    c.end_soc      = sim.end_soc_pct;
    c.heat_kwh     = sim.heat_kwh;
    c.charge_time  = sim.charge_time_s;
    c.charge_power = sim.charge_power_kw;

    /* raw */
    const int raw_ok = sim.end_temp_c >= RAW_T_LOW &&
                       sim.end_temp_c <= RAW_T_HIGH &&
                       sim.end_soc_pct >= RAW_SOC_MIN;
    if (raw_ok) {
        CandSet *s = set_grow(&tls_raw);
        s->v[s->n] = c;
        if (sim.charge_time_s < s->tch_min) s->tch_min = sim.charge_time_s;
        if (sim.charge_time_s > s->tch_max) s->tch_max = sim.charge_time_s;
        if (sim.heat_kwh   < s->e_min)    s->e_min   = sim.heat_kwh;
        if (sim.heat_kwh   > s->e_max)    s->e_max   = sim.heat_kwh;
        ++s->n;
    } else {
        const double pen = ph_constraint_penalty(sim.end_temp_c, sim.end_soc_pct,
                                                 RAW_T_LOW, RAW_T_HIGH,
                                                 RAW_SOC_MIN);
        if (!tls_raw.have_fb || pen < tls_raw.fb_pen - PH_NUM_EPS) {
            tls_raw.fb = c;
            tls_raw.fb_pen = pen;
            tls_raw.have_fb = 1;
        }
    }

    /* safe */
    const int safe_ok = sim.end_temp_c >= SAFE_T_LOW &&
                        sim.end_temp_c <= SAFE_T_HIGH &&
                        sim.end_soc_pct >= SAFE_SOC_MIN;
    if (safe_ok) {
        CandSet *s = set_grow(&tls_safe);
        s->v[s->n] = c;
        if (sim.charge_time_s < s->tch_min) s->tch_min = sim.charge_time_s;
        if (sim.charge_time_s > s->tch_max) s->tch_max = sim.charge_time_s;
        if (sim.heat_kwh   < s->e_min)    s->e_min   = sim.heat_kwh;
        if (sim.heat_kwh   > s->e_max)    s->e_max   = sim.heat_kwh;
        ++s->n;
    }
}

static double score_candidate(double tch, double E,
                              double tch_min, double tch_max,
                              double e_min, double e_max)
{
    const double cr = tch_max - tch_min;
    const double er = e_max - e_min;
    const double s_t = cr > PH_NUM_EPS ? 30.0 * (tch_max - tch) / cr : 30.0;
    const double s_e = er > PH_NUM_EPS ? 20.0 * (e_max - E) / er : 20.0;
    return s_t + s_e;
}

static int cand_cmp_t_asc(const void *a, const void *b)
{
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->t_start < y->t_start) return -1;
    if (x->t_start > y->t_start) return 1;
    return 0;
}

/* score descending; ties -> lower energy; then later start (as in
 * student_solution.c) */
static int cand_cmp_rank(const void *a, const void *b)
{
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->score > y->score + PH_NUM_EPS) return -1;
    if (x->score < y->score - PH_NUM_EPS) return 1;
    if (x->heat_kwh < y->heat_kwh - PH_NUM_EPS) return -1;
    if (x->heat_kwh > y->heat_kwh + PH_NUM_EPS) return 1;
    if (x->t_start > y->t_start) return -1;
    if (x->t_start < y->t_start) return 1;
    return 0;
}

typedef struct {
    int    feasible;
    long   n_eval;          /* filled by caller */
    int    feasible_count;  /* distinct feasible candidates */
    double best_t, best_dist, best_T, best_soc, best_E, best_tch;
    double best_pow, best_score;
    double tch_min, tch_max, e_min, e_max;
    int    n_top;
    Cand   top[TOPK];
    int    have_fb;
    double fb_pen;
    Cand   fb;
} CaseOut;

/* Dedupe (by start time), score, rank. Returns feasibility. */
static void finalize_set(CandSet *s, CaseOut *out)
{
    memset(out, 0, sizeof(*out));
    out->tch_min = s->tch_min; out->tch_max = s->tch_max;
    out->e_min = s->e_min;     out->e_max = s->e_max;
    out->have_fb = s->have_fb; out->fb_pen = s->fb_pen;
    out->fb = s->fb;

    if (s->n <= 0) {
        out->feasible = 0;
        return;
    }

    /* dedupe by t_start (multiple search levels re-hit grid points) */
    qsort(s->v, (size_t)s->n, sizeof(Cand), cand_cmp_t_asc);
    int m = 0;
    for (int i = 0; i < s->n; ++i) {
        if (m == 0 || s->v[i].t_start - s->v[m - 1].t_start > 1.0e-6) {
            s->v[m++] = s->v[i];
        }
    }
    s->n = m;
    out->feasible_count = m;

    for (int i = 0; i < m; ++i) {
        s->v[i].score = score_candidate(s->v[i].charge_time, s->v[i].heat_kwh,
                                        s->tch_min, s->tch_max,
                                        s->e_min, s->e_max);
    }
    qsort(s->v, (size_t)m, sizeof(Cand), cand_cmp_rank);

    const Cand *best = &s->v[0];
    out->feasible = 1;
    out->best_t    = best->t_start;
    out->best_dist = best->dist_km;
    out->best_T    = best->end_temp;
    out->best_soc  = best->end_soc;
    out->best_E    = best->heat_kwh;
    out->best_tch  = best->charge_time;
    out->best_pow  = best->charge_power;
    out->best_score = best->score;

    out->n_top = (m < TOPK) ? m : TOPK;
    for (int i = 0; i < out->n_top; ++i) out->top[i] = s->v[i];
}

/* Span [lo, hi] of feasible start times currently in a set. */
static int set_span(const CandSet *s, double *lo, double *hi)
{
    if (s->n <= 0) return 0;
    *lo = DBL_MAX; *hi = -DBL_MAX;
    for (int i = 0; i < s->n; ++i) {
        if (s->v[i].t_start < *lo) *lo = s->v[i].t_start;
        if (s->v[i].t_start > *hi) *hi = s->v[i].t_start;
    }
    return 1;
}

/* Collect refinement centers: top-K preliminary + interval endpoints. */
static int collect_centers(double *centers, int max_c)
{
    int nc = 0;
    Cand tmp[TOPK];

    /* raw top-K (preliminary scores from current min/max) */
    const CandSet *sets[2] = { &tls_raw, &tls_safe };
    for (int si = 0; si < 2; ++si) {
        const CandSet *s = sets[si];
        if (s->n <= 0) continue;
        int k = 0;
        for (int i = 0; i < s->n; ++i) {
            const double sc = score_candidate(s->v[i].charge_time,
                                              s->v[i].heat_kwh,
                                              s->tch_min, s->tch_max,
                                              s->e_min, s->e_max);
            int pos = k;
            if (k >= TOPK) {
                pos = TOPK;
                for (int j = 0; j < TOPK; ++j) {
                    const double scj = score_candidate(tmp[j].charge_time,
                                                       tmp[j].heat_kwh,
                                                       s->tch_min, s->tch_max,
                                                       s->e_min, s->e_max);
                    int better;
                    if (sc > scj + PH_NUM_EPS) better = 1;
                    else if (sc < scj - PH_NUM_EPS) better = 0;
                    else if (s->v[i].heat_kwh < tmp[j].heat_kwh - PH_NUM_EPS) better = 1;
                    else if (s->v[i].heat_kwh > tmp[j].heat_kwh + PH_NUM_EPS) better = 0;
                    else better = (s->v[i].t_start > tmp[j].t_start);
                    if (better) { pos = j; break; }
                }
            }
            if (pos < TOPK) {
                if (k < TOPK) ++k;
                for (int j = k - 1; j > pos; --j) tmp[j] = tmp[j - 1];
                tmp[pos] = s->v[i];
            }
        }
        for (int j = 0; j < k && nc < max_c; ++j) centers[nc++] = tmp[j].t_start;
    }

    /* interval endpoints */
    double lo, hi;
    if (set_span(&tls_raw, &lo, &hi)) {
        if (nc < max_c) centers[nc++] = lo;
        if (nc < max_c) centers[nc++] = hi;
    }
    if (set_span(&tls_safe, &lo, &hi)) {
        if (nc < max_c) centers[nc++] = lo;
        if (nc < max_c) centers[nc++] = hi;
    }

    /* dedupe */
    for (int i = 0; i < nc; ++i) {
        for (int j = i + 1; j < nc; ++j) {
            if (fabs(centers[j] - centers[i]) < 1.0e-6) {
                centers[j] = centers[nc - 1];
                --nc;
                --j;
            }
        }
    }
    return nc;
}

/* ------------------------------------------------------------------ */
/* Multi-level search for one case                                      */
/* ------------------------------------------------------------------ */
static void search_case(const PhRoute *r, double t0_c, double soc0_pct,
                        double dt, int do_audit, CaseOut *out_raw,
                        CaseOut *out_safe)
{
    /* prefix capacity: steps per route ~ total_time/dt + segment tails */
    const int need = (int)(r->total_time_s / dt) + r->n * 4 + 64;
    ensure_snaps(need);

    PhPrefix pf;
    pf.snaps = tls_snaps;
    pf.cap   = tls_snap_cap;
    pf.n     = 0;
    if (!ph_prefix_build(r, t0_c, soc0_pct, dt, pf.snaps, pf.cap,
                         &pf.n, &pf.end_temp_c, &pf.end_soc_pct)) {
        /* physically invalid case: report infeasible with no fallback */
        set_reset(&tls_raw);
        set_reset(&tls_safe);
        memset(out_raw, 0, sizeof(*out_raw));
        memset(out_safe, 0, sizeof(*out_safe));
        return;
    }

    set_reset(&tls_raw);
    set_reset(&tls_safe);
    long n_eval = 0;
    tls_T_at_zero = -DBL_MAX;

    /* ---- L0: 10 s global coarse sweep ---- */
    for (double t = 0.0; t < r->total_time_s; t += STEP_COARSE_S) {
        evaluate_candidate(r, &pf, dt, t, &n_eval);
    }
    evaluate_candidate(r, &pf, dt, r->total_time_s, &n_eval);

    /* ---- L1: 1 s inside raw windows (or whole route if L0 empty) ----
     *
     * Fast infeasibility proof: T_end is monotonically decreasing in
     * t_start (longer heating -> hotter arrival), so the hottest possible
     * arrival is T_end(0). If even that stays below 20 C, no candidate
     * can ever be feasible -> skip the expensive whole-route sweep. */
    double lo, hi;
    if (set_span(&tls_raw, &lo, &hi)) {
        double w_lo = (lo - STEP_COARSE_S > 0.0) ? lo - STEP_COARSE_S : 0.0;
        double w_hi = (hi + STEP_COARSE_S < r->total_time_s)
                          ? hi + STEP_COARSE_S : r->total_time_s;
        for (double t = w_lo; t <= w_hi + PH_NUM_EPS; t += STEP_MID_S) {
            if (t > r->total_time_s) break;
            evaluate_candidate(r, &pf, dt, t, &n_eval);
        }
    } else if (tls_T_at_zero >= RAW_T_LOW - PH_NUM_EPS) {
        /* narrow feasible window missed by the 10 s grid — full 1 s sweep */
        for (double t = 0.0; t <= r->total_time_s + PH_NUM_EPS; t += STEP_MID_S) {
            if (t > r->total_time_s) t = r->total_time_s;
            evaluate_candidate(r, &pf, dt, t, &n_eval);
            if (t >= r->total_time_s) break;
        }
    }
    /* else: provably infeasible (too cold even starting at t=0) */

    /* ---- L2: 0.1 s around top candidates + interval endpoints ---- */
    double centers[4 * TOPK + 8];
    const int nc = collect_centers(centers, (int)(sizeof(centers) / sizeof(centers[0])));
    for (int i = 0; i < nc; ++i) {
        const double c = centers[i];
        double a = (c - FINE_WINDOW_S > 0.0) ? c - FINE_WINDOW_S : 0.0;
        double b = (c + FINE_WINDOW_S < r->total_time_s)
                       ? c + FINE_WINDOW_S : r->total_time_s;
        for (double t = a; t <= b + PH_NUM_EPS; t += STEP_FINE_S) {
            if (t > r->total_time_s) break;
            evaluate_candidate(r, &pf, dt, t, &n_eval);
        }
    }

    /* ---- L3 (audit only): 0.01 s around current optimum & endpoints ---- */
    if (do_audit) {
        double lo2, hi2;
        if (set_span(&tls_raw, &lo2, &hi2)) {
            double a = (lo2 - AUDIT_WINDOW_S > 0.0) ? lo2 - AUDIT_WINDOW_S : 0.0;
            for (double t = a; t <= lo2 + AUDIT_WINDOW_S; t += STEP_AUDIT_S) {
                evaluate_candidate(r, &pf, dt, t, &n_eval);
            }
            double b = (hi2 + AUDIT_WINDOW_S < r->total_time_s)
                           ? hi2 + AUDIT_WINDOW_S : r->total_time_s;
            for (double t = (hi2 - AUDIT_WINDOW_S > 0.0 ? hi2 - AUDIT_WINDOW_S : 0.0);
                 t <= b; t += STEP_AUDIT_S) {
                evaluate_candidate(r, &pf, dt, t, &n_eval);
            }
        }
        double c = (tls_raw.n > 0) ? tls_raw.v[0].t_start : 0.0;
        if (tls_raw.n > 0) {
            double m_best = -DBL_MAX, t_best = c;
            for (int i = 0; i < tls_raw.n; ++i) {
                const double sc = score_candidate(tls_raw.v[i].charge_time,
                                                  tls_raw.v[i].heat_kwh,
                                                  tls_raw.tch_min, tls_raw.tch_max,
                                                  tls_raw.e_min, tls_raw.e_max);
                if (sc > m_best) { m_best = sc; t_best = tls_raw.v[i].t_start; }
            }
            double a = (t_best - AUDIT_WINDOW_S > 0.0) ? t_best - AUDIT_WINDOW_S : 0.0;
            double b = (t_best + AUDIT_WINDOW_S < r->total_time_s)
                           ? t_best + AUDIT_WINDOW_S : r->total_time_s;
            for (double t = a; t <= b + PH_NUM_EPS; t += STEP_AUDIT_S) {
                evaluate_candidate(r, &pf, dt, t, &n_eval);
            }
        }
    }

    finalize_set(&tls_raw, out_raw);
    finalize_set(&tls_safe, out_safe);
    out_raw->n_eval = n_eval;
    out_safe->n_eval = n_eval;
}

/* ------------------------------------------------------------------ */
/* Selftest: replicate student_solution.c on the official mock route    */
/* (1 s candidate grid, dt = 0.5, raw constraints, same tie-breaks)     */
/* ------------------------------------------------------------------ */
static int run_selftest(void)
{
    PhSeg segs[6] = {
        {2.0, 30.0, 15.0, -8.0},
        {5.0, 60.0, 22.0, -9.0},
        {15.0, 90.0, 30.0, -10.0},
        {12.0, 100.0, 35.0, -10.0},
        {3.0, 40.0, 18.0, -9.0},
        {2.0, 20.0, 12.0, -8.0},
    };
    PhRoute r;
    if (!ph_route_build(&r, segs, 6)) {
        printf("SELFTEST FAIL: bad mock route\n");
        return 1;
    }
    const double dt = 0.5;
    const int need = (int)(r.total_time_s / dt) + 32;
    ensure_snaps(need);
    PhPrefix pf;
    pf.snaps = tls_snaps; pf.cap = tls_snap_cap; pf.n = 0;
    if (!ph_prefix_build(&r, 0.0, 65.0, dt, pf.snaps, pf.cap,
                         &pf.n, &pf.end_temp_c, &pf.end_soc_pct)) {
        printf("SELFTEST FAIL: prefix build failed\n");
        return 1;
    }

    set_reset(&tls_raw);
    long n_eval = 0;
    for (long idx = 0; ; ++idx) {
        double t = (double)idx * 1.0;
        if (t > r.total_time_s) t = r.total_time_s;
        evaluate_candidate(&r, &pf, dt, t, &n_eval);
        if (t >= r.total_time_s - PH_NUM_EPS) break;
    }

    CaseOut out;
    finalize_set(&tls_raw, &out);

    printf("=== Oracle kernel selftest vs official mock baseline ===\n");
    printf("  total_time_s      : %.6f\n", r.total_time_s);
    printf("  n_evaluated       : %ld\n", n_eval);
    printf("  feasible_count    : %d   (expected 342)\n", out.feasible_count);
    printf("  best_start_time_s : %.6f\n", out.best_t);
    printf("  best_start_dist_km: %.6f   (expected 35.32)\n", out.best_dist);
    printf("  T_end             : %.6f   (expected 20.00)\n", out.best_T);
    printf("  SOC_end           : %.6f   (expected 15.95)\n", out.best_soc);
    printf("  E_heat_kWh        : %.6f   (expected 3.1017)\n", out.best_E);
    printf("  charge_time_s     : %.6f   (expected 947.19)\n", out.best_tch);
    printf("  score             : %.6f   (expected 50.000)\n", out.best_score);

    int pass = 1;
    if (out.feasible_count != 342) pass = 0;
    if (fabs(out.best_dist - 35.32) > 0.01) pass = 0;
    if (fabs(out.best_T - 20.00) > 0.01) pass = 0;
    if (fabs(out.best_soc - 15.95) > 0.01) pass = 0;
    if (fabs(out.best_E - 3.1017) > 0.001) pass = 0;
    if (fabs(out.best_tch - 947.19) > 0.05) pass = 0;
    if (fabs(out.best_score - 50.0) > 0.01) pass = 0;

    printf("  ==> %s\n", pass ? "GATE PASSED" : "GATE FAILED");
    return pass ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* dt convergence study                                                 */
/* ------------------------------------------------------------------ */
static int ensure_parent_dir(const char *prefix)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", prefix);
    for (char *p = buf + 1; *p; ++p) {
        if (*p == '/' || *p == '\\') {
            const char c = *p;
            *p = '\0';
            MKDIR(buf); /* ignore EEXIST */
            *p = c;
        }
    }
    return 0;
}

static int run_dt_study(const char *path, long n_samples, unsigned seed)
{
    ensure_parent_dir("output/convergence/");
    if (parse_dataset(path) < 0) return 1;
    if (g_n_cases <= 0) { fprintf(stderr, "no cases\n"); return 1; }

    const double dts[4] = {1.0, 0.5, 0.25, 0.125};
    const int ND = 4;

    long *idx = (long *)malloc((size_t)n_samples * sizeof(long));
    unsigned x = seed;
    for (long i = 0; i < n_samples; ++i) {
        x = x * 1103515245u + 12345u;
        idx[i] = (long)((x >> 16) % (unsigned)g_n_cases);
    }

    typedef struct { double t, T, soc, score; int feas; } R;
    R *res = (R *)calloc((size_t)n_samples * ND, sizeof(R));

    double t0 = (double)clock();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (long i = 0; i < n_samples; ++i) {
        PhRoute r;
        const CaseData *cd = &g_cases[idx[i]];
        if (!ph_route_build(&r, cd->segs, cd->n_segs)) continue;
        for (int d = 0; d < ND; ++d) {
            CaseOut oraw, osafe;
            search_case(&r, 0.0, 65.0, dts[d], 0, &oraw, &osafe);
            R *rr = &res[i * ND + d];
            rr->feas = oraw.feasible;
            rr->t = oraw.best_t;
            rr->T = oraw.best_T;
            rr->soc = oraw.best_soc;
            rr->score = oraw.best_score;
        }
    }
    const double elapsed = ((double)clock() - t0) / CLOCKS_PER_SEC;

    FILE *f = fopen("output/convergence/dt_study.csv", "w");
    if (!f) { fprintf(stderr, "cannot write dt_study.csv\n"); return 1; }
    fprintf(f, "case_id");
    for (int d = 0; d < ND; ++d) fprintf(f, ",t_best_dt%.3g,T_dt%.3g,soc_dt%.3g,score_dt%.3g", dts[d], dts[d], dts[d], dts[d]);
    fprintf(f, ",dT_1.0,dT_0.5,dT_0.25,dSoc_1.0,dSoc_0.5,dSoc_0.25,dt_1.0,dt_0.5,dt_0.25,dscore_1.0,dscore_0.5,dscore_0.25\n");

    double sum_dT[3] = {0}, max_dT[3] = {0};
    double sum_dS[3] = {0}, max_dS[3] = {0};
    double sum_dt_[3] = {0}, max_dt_[3] = {0};
    double sum_dsc[3] = {0}, max_dsc[3] = {0};
    long n_both = 0;

    for (long i = 0; i < n_samples; ++i) {
        const R *base = &res[i * ND + 3]; /* dt = 0.125 */
        fprintf(f, "%ld", idx[i] + 1);
        for (int d = 0; d < ND; ++d) {
            const R *rr = &res[i * ND + d];
            fprintf(f, ",%.4f,%.6f,%.6f,%.6f", rr->t, rr->T, rr->soc, rr->score);
        }
        if (base->feas) {
            int allf = 1;
            for (int d = 0; d < 3; ++d) if (!res[i * ND + d].feas) allf = 0;
            if (allf) {
                ++n_both;
                for (int d = 0; d < 3; ++d) {
                    const R *rr = &res[i * ND + d];
                    const double aT = fabs(rr->T - base->T);
                    const double aS = fabs(rr->soc - base->soc);
                    const double at = fabs(rr->t - base->t);
                    const double asc = fabs(rr->score - base->score);
                    sum_dT[d] += aT;  if (aT > max_dT[d]) max_dT[d] = aT;
                    sum_dS[d] += aS;  if (aS > max_dS[d]) max_dS[d] = aS;
                    sum_dt_[d] += at; if (at > max_dt_[d]) max_dt_[d] = at;
                    sum_dsc[d] += asc; if (asc > max_dsc[d]) max_dsc[d] = asc;
                    fprintf(f, ",%.6g", aT);
                }
                for (int d = 0; d < 3; ++d) fprintf(f, ",%.6g", fabs(res[i*ND+d].soc - base->soc));
                for (int d = 0; d < 3; ++d) fprintf(f, ",%.6g", fabs(res[i*ND+d].t - base->t));
                for (int d = 0; d < 3; ++d) fprintf(f, ",%.6g", fabs(res[i*ND+d].score - base->score));
            } else {
                for (int k = 0; k < 12; ++k) fprintf(f, ",");
            }
        } else {
            for (int k = 0; k < 12; ++k) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);

    printf("\n=== dt convergence summary (baseline dt=0.125, %ld cases) ===\n", n_both);
    printf("%-10s %-12s %-12s %-12s %-12s\n", "dt", "mean|dT|", "max|dT|", "mean|dt_start|", "max|dt_start|");
    for (int d = 0; d < 3; ++d) {
        printf("%-10g %-12.3g %-12.3g %-12.3g %-12.3g\n", dts[d],
               sum_dT[d] / (n_both ? n_both : 1), max_dT[d],
               sum_dt_[d] / (n_both ? n_both : 1), max_dt_[d]);
    }
    printf("%-10s %-12s %-12s %-12s %-12s\n", "dt", "mean|dSOC|", "max|dSOC|", "mean|dscore|", "max|dscore|");
    for (int d = 0; d < 3; ++d) {
        printf("%-10g %-12.3g %-12.3g %-12.3g %-12.3g\n", dts[d],
               sum_dS[d] / (n_both ? n_both : 1), max_dS[d],
               sum_dsc[d] / (n_both ? n_both : 1), max_dsc[d]);
    }
    printf("total CPU time: %.1f s\n", elapsed);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Batch generation                                                     */
/* ------------------------------------------------------------------ */
static void write_case_row(FILE *f, long id, const CaseOut *o)
{
    if (o->feasible) {
        fprintf(f, "%ld,1,%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.4f,%.6f,%d,"
                   "%.6f,%.6f,%.6f,%.6f,%ld,0\n",
                id, o->best_t, o->best_dist, o->best_T, o->best_soc,
                o->best_E, o->best_tch, o->best_pow, o->best_score,
                o->feasible_count, o->tch_min, o->tch_max,
                o->e_min, o->e_max, o->n_eval);
    } else if (o->have_fb) {
        fprintf(f, "%ld,0,%.4f,%.4f,%.6f,%.6f,%.6f,%.4f,%.4f,-1,0,,,,,%ld,%.6g\n",
                id, o->fb.t_start, o->fb.dist_km, o->fb.end_temp,
                o->fb.end_soc, o->fb.heat_kwh, o->fb.charge_time,
                o->fb.charge_power, o->n_eval, o->fb_pen);
    } else {
        fprintf(f, "%ld,0,,,,,,,,,,,,,%ld,\n", id, o->n_eval);
    }
}

static void write_top_rows(FILE *f, long id, const CaseOut *o)
{
    for (int i = 0; i < o->n_top; ++i) {
        const Cand *c = &o->top[i];
        fprintf(f, "%ld,%d,%.4f,%.4f,%.6f,%.6f,%.6f,%.4f,%.6f\n",
                id, i + 1, c->t_start, c->dist_km, c->end_temp, c->end_soc,
                c->heat_kwh, c->charge_time, c->score);
    }
}

static const char *MAIN_HEADER =
    "case_id,feasible,best_start_time_s,best_start_distance_km,T_end,"
    "SOC_end_pct,E_heat_kwh,charge_time_s,charge_power_kw,score,"
    "feasible_count,t_charge_min,t_charge_max,E_heat_min,E_heat_max,"
    "n_eval,fb_penalty\n";
static const char *TOP_HEADER =
    "case_id,rank,start_time_s,start_distance_km,T_end,SOC_end_pct,"
    "E_heat_kwh,charge_time_s,score\n";

static long find_last_case_id(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[512];
    long last = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] >= '0' && line[0] <= '9') {
            last = strtol(line, NULL, 10);
        }
    }
    fclose(f);
    return last;
}

static int run_generate(const char *path, double dt, long start, long end,
                        int resume, const char *out_prefix, long audit_n)
{
    ensure_parent_dir(out_prefix);
    if (parse_dataset(path) < 0) return 1;
    if (g_n_cases <= 0) { fprintf(stderr, "no cases parsed\n"); return 1; }
    if (end <= 0 || end > g_n_cases) end = g_n_cases;
    if (start <= 0) start = 1;
    if (start > end) { fprintf(stderr, "empty range\n"); return 1; }

    char p_raw[512], p_safe[512], p_top_raw[512], p_top_safe[512], p_audit[512];
    snprintf(p_raw, sizeof(p_raw), "%s_raw.csv", out_prefix);
    snprintf(p_safe, sizeof(p_safe), "%s_safe.csv", out_prefix);
    snprintf(p_top_raw, sizeof(p_top_raw), "%s_top10_raw.csv", out_prefix);
    snprintf(p_top_safe, sizeof(p_top_safe), "%s_top10_safe.csv", out_prefix);
    snprintf(p_audit, sizeof(p_audit), "%s_audit.csv", out_prefix);

    if (resume) {
        const long last = find_last_case_id(p_raw);
        if (last >= start) {
            printf("resume: last case_id=%ld in %s -> continuing from %ld\n",
                   last, p_raw, last + 1);
            start = last + 1;
        }
    }

    const int fresh = (start == 1);
    FILE *f_raw = fopen(p_raw, fresh ? "w" : "a");
    FILE *f_safe = fopen(p_safe, fresh ? "w" : "a");
    FILE *f_tr = fopen(p_top_raw, fresh ? "w" : "a");
    FILE *f_ts = fopen(p_top_safe, fresh ? "w" : "a");
    if (!f_raw || !f_safe || !f_tr || !f_ts) {
        fprintf(stderr, "cannot open output files (prefix %s)\n", out_prefix);
        return 1;
    }
    if (fresh) {
        fprintf(f_raw, "%s", MAIN_HEADER);
        fprintf(f_safe, "%s", MAIN_HEADER);
        fprintf(f_tr, "%s", TOP_HEADER);
        fprintf(f_ts, "%s", TOP_HEADER);
    }
    FILE *f_au = NULL;
    if (audit_n > 0) {
        f_au = fopen(p_audit, fresh ? "w" : "a");
        if (f_au) fprintf(f_au, "case_id,best_t_normal,best_t_audited,shift_s\n");
    }

    const long chunk = 2000;
    clock_t t_begin = clock();
    long done = 0;

    for (long base = start; base <= end; base += chunk) {
        const long hi = (base + chunk - 1 < end) ? base + chunk - 1 : end;
        const long cnt = hi - base + 1;
        CaseOut *rr = (CaseOut *)malloc((size_t)cnt * sizeof(CaseOut));
        CaseOut *rs = (CaseOut *)malloc((size_t)cnt * sizeof(CaseOut));
        CaseOut *ar = NULL, *as_ = NULL;
        const long n_audit = (audit_n > 0 && base == start && audit_n <= cnt)
                                 ? audit_n : 0;
        if (n_audit > 0) {
            ar = (CaseOut *)malloc((size_t)n_audit * sizeof(CaseOut));
            as_ = (CaseOut *)malloc((size_t)n_audit * sizeof(CaseOut));
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (long i = 0; i < cnt; ++i) {
            const long id = base + i;
            PhRoute r;
            if (!ph_route_build(&r, g_cases[id - 1].segs,
                                g_cases[id - 1].n_segs)) {
                memset(&rr[i], 0, sizeof(rr[i]));
                memset(&rs[i], 0, sizeof(rs[i]));
                continue;
            }
            search_case(&r, 0.0, 65.0, dt, 0, &rr[i], &rs[i]);
            if (n_audit > 0 && i < n_audit) {
                search_case(&r, 0.0, 65.0, dt, 1, &ar[i], &as_[i]);
            }
        }

        for (long i = 0; i < cnt; ++i) {
            const long id = base + i;
            write_case_row(f_raw, id, &rr[i]);
            write_case_row(f_safe, id, &rs[i]);
            write_top_rows(f_tr, id, &rr[i]);
            write_top_rows(f_ts, id, &rs[i]);
        }
        if (n_audit > 0) {
            for (long i = 0; i < n_audit; ++i) {
                const double shift = (rr[i].feasible && ar[i].feasible)
                    ? fabs(rr[i].best_t - ar[i].best_t) : -1.0;
                fprintf(f_au, "%ld,%.4f,%.4f,%.4f\n", base + i,
                        rr[i].best_t, ar[i].best_t, shift);
            }
        }
        fflush(f_raw); fflush(f_safe); fflush(f_tr); fflush(f_ts);
        if (f_au) fflush(f_au);

        free(rr); free(rs); free(ar); free(as_);
        done += cnt;
        const double el = (double)(clock() - t_begin) / CLOCKS_PER_SEC;
        printf("progress: %ld / %ld  (%.1f%%)  cpu=%.0fs\n",
               done, end - start + 1, 100.0 * done / (end - start + 1), el);
    }

    fclose(f_raw); fclose(f_safe); fclose(f_tr); fclose(f_ts);
    if (f_au) fclose(f_au);
    printf("done. outputs at prefix: %s\n", out_prefix);
    return 0;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *input = "../sdk/data/nav_train_100000.txt";
    const char *out_prefix = "output/raw_oracle_prefix";
    const char *mode = NULL;
    double dt = 0.5;
    long start = 1, end = 0, audit = 0, n_study = 500;
    unsigned seed = 42;
    int resume = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selftest") == 0) mode = "selftest";
        else if (strcmp(argv[i], "--dt-study") == 0) mode = "dtstudy";
        else if (strcmp(argv[i], "--generate") == 0) mode = "generate";
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
        else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) out_prefix = argv[++i];
        else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) dt = atof(argv[++i]);
        else if (strcmp(argv[i], "--start") == 0 && i + 1 < argc) start = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--end") == 0 && i + 1 < argc) end = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--audit") == 0 && i + 1 < argc) audit = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n_study = strtol(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--resume") == 0) resume = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }

    if (mode == NULL) {
        fprintf(stderr,
                "usage: oracle_generator --selftest\n"
                "       oracle_generator --dt-study [--n 500] [--seed 42] [--input FILE]\n"
                "       oracle_generator --generate [--dt 0.5] [--start 1] [--end 100000]\n"
                "                               [--resume] [--audit 200]\n"
                "                               [--out-prefix output/oracle]\n");
        return 1;
    }
    if (strcmp(mode, "selftest") == 0) return run_selftest();
    if (strcmp(mode, "dtstudy") == 0) return run_dt_study(input, n_study, seed);
    return run_generate(input, dt, start, end, resume, out_prefix, audit);
}
