#include "../include/seat_sdk.h"
#include <math.h>
#include <string.h>

typedef struct {
    float data[BASELINE_WINDOW];
    int head;
    int count;
} RingBuffer;

typedef struct {
    float mean;
    float sigma;
} BaselineResult;

typedef struct {
    RingBuffer fwd_curr_buf;
    RingBuffer fwd_hw_buf;
    RingBuffer rev_curr_buf;
    RingBuffer rev_hw_buf;

    float curr_baseline;
    float hw_baseline;
    float curr_sigma;
    float hw_sigma;

    OperMotSt_u8 prev_oper_st;
    Posn_u8 prev_posn;
    int startup_counter;
    int abnormal_frames;
    int hall_evidence_hold;
    bool pinch_latched;
} DetectorState;

/*
 * Hall pulse width is updated only when a new Hall edge arrives.  Between two
 * edges the input can return to its normal level, so requiring current and
 * pulse width to be high on every single scheduler call causes false resets.
 * Keep a valid Hall slowdown observation for a few calls while still requiring
 * the motor-current anomaly to persist for MIN_DURATION_SAMPLES.
 */
#define HALL_EVIDENCE_HOLD_SAMPLES  3
#define HW_EFFECTIVE_SIGMA_FACTOR   2.0f

static DetectorState g_det;

static void ringbuf_init(RingBuffer *rb)
{
    rb->head = 0;
    rb->count = 0;
}

static void ringbuf_clear(RingBuffer *rb)
{
    rb->head = 0;
    rb->count = 0;
}

static void ringbuf_push(RingBuffer *rb, float value)
{
    rb->data[rb->head] = value;
    rb->head = (rb->head + 1) % BASELINE_WINDOW;
    if (rb->count < BASELINE_WINDOW) {
        rb->count++;
    }
}

static BaselineResult ringbuf_calc_baseline(const RingBuffer *rb, float fallback)
{
    BaselineResult result;

    if (rb->count <= 0) {
        result.mean = fallback;
        result.sigma = 0.0f;
        return result;
    }

    double sum = 0.0;
    int start = (rb->count < BASELINE_WINDOW) ? 0 : rb->head;
    for (int i = 0; i < rb->count; ++i) {
        int idx = (start + i) % BASELINE_WINDOW;
        sum += (double)rb->data[idx];
    }

    double mean = sum / (double)rb->count;
    result.mean = (float)mean;

    if (rb->count < 2) {
        result.sigma = 0.0f;
        return result;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < rb->count; ++i) {
        int idx = (start + i) % BASELINE_WINDOW;
        double diff = (double)rb->data[idx] - mean;
        sum_sq += diff * diff;
    }

    result.sigma = (float)sqrt(sum_sq / (double)rb->count);
    return result;
}

static bool motor_is_running(OperMotSt_u8 oper_st)
{
    return oper_st == OPERMOTST_LEFT_UP || oper_st == OPERMOTST_RIGHT_DOWN;
}

static bool in_pinch_zone(Posn_u8 posn, OperMotSt_u8 oper_st)
{
    if (!motor_is_running(oper_st)) {
        return false;
    }
    return posn >= PINCH_ZONE_MIN && posn <= PINCH_ZONE_MAX;
}

static RingBuffer *current_curr_buffer(OperMotSt_u8 oper_st)
{
    return (oper_st == OPERMOTST_LEFT_UP) ? &g_det.fwd_curr_buf : &g_det.rev_curr_buf;
}

static RingBuffer *current_hw_buffer(OperMotSt_u8 oper_st)
{
    return (oper_st == OPERMOTST_LEFT_UP) ? &g_det.fwd_hw_buf : &g_det.rev_hw_buf;
}

void anti_pinch_detector_init(void)
{
    memset(&g_det, 0, sizeof(g_det));
    ringbuf_init(&g_det.fwd_curr_buf);
    ringbuf_init(&g_det.fwd_hw_buf);
    ringbuf_init(&g_det.rev_curr_buf);
    ringbuf_init(&g_det.rev_hw_buf);

    g_det.prev_oper_st = OPERMOTST_NO_RUNNING;
    g_det.prev_posn = 0xFF;
    g_det.startup_counter = 0;
    g_det.abnormal_frames = 0;
    g_det.hall_evidence_hold = 0;
    g_det.pinch_latched = false;
    g_det.curr_baseline = 0.0f;
    g_det.hw_baseline = 0.0f;
    g_det.curr_sigma = 0.0f;
    g_det.hw_sigma = 0.0f;
}

void anti_pinch_detector_step(void)
{
    MotCurr_u16 mot_curr_val = 0;
    HallPosn_u16 hall_posn_val = 0;
    HallPlsWidth_u32 hall_width_val = 0;
    OperMotSt_u8 oper_st_val = OPERMOTST_NO_RUNNING;
    MotPwrVolt_u16 batt_vol_val = 0;
    Posn_u8 axis_posn_val = 0xFF;

    SeatBackRclnMotDD_u16MotCur(&mot_curr_val);
    BO_Atm_SeatBackRcln_ntfHallPosn(&hall_posn_val);
    SeatBackRclnHallDD_u32CurrHallPlsWidth(&hall_width_val);
    BO_Atm_SeatBackRcln_ntfOperSt(&oper_st_val);
    SeatBackRclnMotDD_u16MotPwrVolt(&batt_vol_val);
    BO_Atm_SeatBackRcln_ntfPosn(&axis_posn_val);

    (void)hall_posn_val;

    bool motor_running = motor_is_running(oper_st_val);
    bool rising_edge = motor_running && !motor_is_running(g_det.prev_oper_st);
    bool falling_edge = !motor_running && motor_is_running(g_det.prev_oper_st);
    bool direction_changed = motor_running && motor_is_running(g_det.prev_oper_st)
                           && oper_st_val != g_det.prev_oper_st;

    if (rising_edge || direction_changed) {
        g_det.startup_counter = STARTUP_DELAY_SAMPLES;
        g_det.abnormal_frames = 0;
        g_det.hall_evidence_hold = 0;
        g_det.pinch_latched = false;

        RingBuffer *curr_buf = current_curr_buffer(oper_st_val);
        RingBuffer *hw_buf = current_hw_buffer(oper_st_val);
        ringbuf_clear(curr_buf);
        ringbuf_clear(hw_buf);
    }

    if (motor_running && g_det.startup_counter > 0) {
        g_det.startup_counter--;
    }

    bool in_startup = g_det.startup_counter > 0;

    float batt_vol = (float)batt_vol_val * V_ADC_FACTOR;
    float curr_norm = (batt_vol > 0.1f) ? ((float)mot_curr_val * V_REF / batt_vol) : (float)mot_curr_val;
    float hw_val = (float)hall_width_val;

    bool pinch_now = false;
    if (motor_running && !in_startup && !g_det.pinch_latched) {
        RingBuffer *curr_buf = current_curr_buffer(oper_st_val);
        RingBuffer *hw_buf = current_hw_buffer(oper_st_val);

        /* Calculate against history first; never let the current sample hide
         * its own deviation by pulling the baseline toward itself. */
        BaselineResult curr_bl = ringbuf_calc_baseline(curr_buf, curr_norm);
        BaselineResult hw_bl = ringbuf_calc_baseline(hw_buf, hw_val);

        g_det.curr_baseline = curr_bl.mean;
        g_det.hw_baseline = hw_bl.mean;
        g_det.curr_sigma = curr_bl.sigma;
        g_det.hw_sigma = hw_bl.sigma;

        bool enough_data = (curr_buf->count >= BASELINE_MIN_FILL) && (hw_buf->count >= BASELINE_MIN_FILL);
        bool freeze_baseline = false;

        if (enough_data && in_pinch_zone(axis_posn_val, oper_st_val)) {
            float curr_thresh = fmaxf(g_det.curr_sigma * (float)CURR_SIGMA_FACTOR, (float)CURR_DEV_MIN);
            /* Hall width is a pulsed signal rather than Gaussian noise.  A
             * three-sigma limit is inflated by its normal high/low cadence and
             * recognizes the slowdown only just before the motor stops.  The
             * fixed floor, current-channel gate and duration filter still
             * provide three independent guards against a single noisy edge. */
            float hw_thresh = fmaxf(g_det.hw_sigma * HW_EFFECTIVE_SIGMA_FACTOR, (float)HW_DEV_MIN);
            float curr_dev = curr_norm - g_det.curr_baseline;
            float hw_dev = hw_val - g_det.hw_baseline;

            bool curr_high = curr_dev > curr_thresh;
            bool hall_slow = hw_dev > hw_thresh;

            if (hall_slow) {
                g_det.hall_evidence_hold = HALL_EVIDENCE_HOLD_SAMPLES;
            } else if (g_det.hall_evidence_hold > 0) {
                g_det.hall_evidence_hold--;
            }

            if (curr_high && g_det.hall_evidence_hold > 0) {
                g_det.abnormal_frames++;
                freeze_baseline = true;
            } else {
                g_det.abnormal_frames = 0;
            }

            if (g_det.abnormal_frames >= MIN_DURATION_SAMPLES) {
                pinch_now = true;
                g_det.pinch_latched = true;
            }
        } else {
            g_det.abnormal_frames = 0;
            g_det.hall_evidence_hold = 0;
        }

        /* Learn only normal operation.  Once both signals form a candidate,
         * hold the pre-event baseline until the candidate is confirmed or
         * rejected. */
        if (!freeze_baseline && !g_det.pinch_latched) {
            ringbuf_push(curr_buf, curr_norm);
            ringbuf_push(hw_buf, hw_val);
        }
    } else {
        g_det.abnormal_frames = 0;
        g_det.hall_evidence_hold = 0;
    }

    if (falling_edge) {
        g_det.pinch_latched = false;
        g_det.abnormal_frames = 0;
        g_det.hall_evidence_hold = 0;
    }

    g_det.prev_oper_st = oper_st_val;
    g_det.prev_posn = axis_posn_val;

    BO_Atm_SeatBackRcln_ntfAntiPinchSt(pinch_now || g_det.pinch_latched ? ANTIPINCHST_OCCURRED : ANTIPINCHST_NORMAL);
}
