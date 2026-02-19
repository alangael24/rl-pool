#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"

#define PI 3.14159265358979323846f
#define NUM_SHOT_DIRS 16
#define NUM_POWER_BINS 5
#define STATIONARY_EPS 0.0005f

#define NUM_PLAYERS 2
#define NUM_BALLS 16
#define NUM_POCKETS 6
#define OBS_SIZE 89

#define CUE_BALL 0
#define EIGHT_BALL 8

#define GROUP_UNKNOWN 0
#define GROUP_SOLIDS 1
#define GROUP_STRIPES 2

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float shots;
    float n;
};

typedef struct Client Client;
struct Client {
    int width_px;
    int height_px;
    int margin;
    float scale;
};

typedef struct Pool Pool;
struct Pool {
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;

    Log log;
    Client* client;

    float table_width;
    float table_height;
    float ball_radius;
    float pocket_radius;
    float friction;
    float restitution;
    float impulse;
    float min_power;

    float reward_step;
    float reward_shot;
    float reward_legal_pot;
    float reward_illegal_pot;
    float reward_foul;
    float reward_win;

    unsigned char fast_forward;
    int max_physics_steps;
    int max_steps;

    float inv_table_width;
    float inv_table_height;
    float inv_vel_scale;
    float inv_max_steps;
    float pocket_radius_sq;

    float shot_cos[NUM_SHOT_DIRS];
    float shot_sin[NUM_SHOT_DIRS];
    float shot_impulses[NUM_POWER_BINS];

    float pocket_x[NUM_POCKETS];
    float pocket_y[NUM_POCKETS];

    float ball_x[NUM_BALLS];
    float ball_y[NUM_BALLS];
    float ball_vx[NUM_BALLS];
    float ball_vy[NUM_BALLS];
    unsigned char ball_alive[NUM_BALLS];

    int player_group[NUM_PLAYERS];
    int current_player;
    unsigned char ball_in_hand;

    int tick;
    float episode_return[NUM_PLAYERS];
    int shots_taken[NUM_PLAYERS];
    int fouls[NUM_PLAYERS];
    int winner;
};

static inline float randf(float lo, float hi) {
    return lo + ((float)rand() / (float)RAND_MAX) * (hi - lo);
}

static inline float dist_sq(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return dx * dx + dy * dy;
}

static inline int group_for_ball(int idx) {
    if (idx >= 1 && idx <= 7) return GROUP_SOLIDS;
    if (idx >= 9 && idx <= 15) return GROUP_STRIPES;
    return GROUP_UNKNOWN;
}

static inline bool balls_stationary(Pool* env) {
    float eps2 = STATIONARY_EPS * STATIONARY_EPS;
    for (int i = 0; i < NUM_BALLS; i++) {
        if (!env->ball_alive[i]) continue;
        float s2 = env->ball_vx[i] * env->ball_vx[i] + env->ball_vy[i] * env->ball_vy[i];
        if (s2 >= eps2) return false;
    }
    return true;
}

static inline void shuffle_ints(int* arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static inline int count_remaining_group(Pool* env, int group) {
    if (group == GROUP_UNKNOWN) return 0;

    int count = 0;
    for (int i = 1; i < NUM_BALLS; i++) {
        if (!env->ball_alive[i]) continue;
        if (group_for_ball(i) == group) count++;
    }
    return count;
}

static inline const char* group_name(int g) {
    if (g == GROUP_SOLIDS) return "solids";
    if (g == GROUP_STRIPES) return "stripes";
    return "open";
}

static inline void set_pockets(Pool* env) {
    float px = env->pocket_radius * 1.1f;
    float py = env->pocket_radius * 1.1f;

    env->pocket_x[0] = px;
    env->pocket_y[0] = py;

    env->pocket_x[1] = env->table_width * 0.5f;
    env->pocket_y[1] = py;

    env->pocket_x[2] = env->table_width - px;
    env->pocket_y[2] = py;

    env->pocket_x[3] = px;
    env->pocket_y[3] = env->table_height - py;

    env->pocket_x[4] = env->table_width * 0.5f;
    env->pocket_y[4] = env->table_height - py;

    env->pocket_x[5] = env->table_width - px;
    env->pocket_y[5] = env->table_height - py;
}

static inline bool in_any_pocket_zone(Pool* env, float x, float y, float pad) {
    float pocket_zone = (env->pocket_radius + pad) * (env->pocket_radius + pad);
    for (int p = 0; p < NUM_POCKETS; p++) {
        if (dist_sq(x, y, env->pocket_x[p], env->pocket_y[p]) < pocket_zone) {
            return true;
        }
    }
    return false;
}

static inline void place_cue_ball_in_hand(Pool* env) {
    float margin = env->ball_radius * 2.5f;
    float min_dist = env->ball_radius * 2.1f;
    float min_dist_sq = min_dist * min_dist;

    for (int i = 0; i < 512; i++) {
        float x = randf(margin, env->table_width * 0.45f);
        float y = randf(margin, env->table_height - margin);

        if (in_any_pocket_zone(env, x, y, env->ball_radius)) {
            continue;
        }

        bool overlap = false;
        for (int b = 1; b < NUM_BALLS; b++) {
            if (!env->ball_alive[b]) continue;
            if (dist_sq(x, y, env->ball_x[b], env->ball_y[b]) < min_dist_sq) {
                overlap = true;
                break;
            }
        }

        if (!overlap) {
            env->ball_alive[CUE_BALL] = 1;
            env->ball_x[CUE_BALL] = x;
            env->ball_y[CUE_BALL] = y;
            env->ball_vx[CUE_BALL] = 0.0f;
            env->ball_vy[CUE_BALL] = 0.0f;
            return;
        }
    }

    env->ball_alive[CUE_BALL] = 1;
    env->ball_x[CUE_BALL] = env->table_width * 0.25f;
    env->ball_y[CUE_BALL] = env->table_height * 0.50f;
    env->ball_vx[CUE_BALL] = 0.0f;
    env->ball_vy[CUE_BALL] = 0.0f;
}

static inline void rack_balls(Pool* env) {
    for (int i = 0; i < NUM_BALLS; i++) {
        env->ball_alive[i] = 1;
        env->ball_vx[i] = 0.0f;
        env->ball_vy[i] = 0.0f;
        env->ball_x[i] = 0.0f;
        env->ball_y[i] = 0.0f;
    }

    place_cue_ball_in_hand(env);
    env->ball_in_hand = 0;

    int rack_ids[15];
    int rem[14];
    int rem_idx = 0;
    for (int i = 1; i < NUM_BALLS; i++) {
        if (i == EIGHT_BALL) continue;
        rem[rem_idx++] = i;
    }
    shuffle_ints(rem, 14);

    int p = 0;
    for (int i = 0; i < 15; i++) {
        if (i == 4) {
            rack_ids[i] = EIGHT_BALL;
        } else {
            rack_ids[i] = rem[p++];
        }
    }

    float rack_x = env->table_width * 0.70f;
    float rack_y = env->table_height * 0.50f;
    float dx = env->ball_radius * 1.95f;
    float dy = env->ball_radius * 2.05f;

    int slot = 0;
    for (int row = 0; row < 5; row++) {
        for (int k = 0; k <= row; k++) {
            int ball = rack_ids[slot++];
            env->ball_x[ball] = rack_x + row * dx;
            env->ball_y[ball] = rack_y + (k - row * 0.5f) * dy;
            env->ball_vx[ball] = 0.0f;
            env->ball_vy[ball] = 0.0f;
            env->ball_alive[ball] = 1;
        }
    }
}

static inline void update_derived_constants(Pool* env) {
    float width = fmaxf(env->table_width, 1e-6f);
    float height = fmaxf(env->table_height, 1e-6f);
    float vel_scale = fmaxf(env->impulse, 1e-3f);

    env->inv_table_width = 1.0f / width;
    env->inv_table_height = 1.0f / height;
    env->inv_vel_scale = 1.0f / vel_scale;
    env->inv_max_steps = 1.0f / fmaxf((float)env->max_steps, 1.0f);
    env->pocket_radius_sq = env->pocket_radius * env->pocket_radius;

    for (int i = 0; i < NUM_SHOT_DIRS; i++) {
        float angle = (2.0f * PI * (float)i) / (float)NUM_SHOT_DIRS;
        env->shot_cos[i] = cosf(angle);
        env->shot_sin[i] = sinf(angle);
    }

    for (int i = 0; i < NUM_POWER_BINS; i++) {
        float power_frac = (float)i / (float)(NUM_POWER_BINS - 1);
        float power_scale = env->min_power + (1.0f - env->min_power) * power_frac;
        env->shot_impulses[i] = env->impulse * power_scale;
    }

    set_pockets(env);
}

static inline void compute_single_observation(Pool* env, int player, float* obs) {
    int idx = 0;
    for (int i = 0; i < NUM_BALLS; i++) {
        float alive = env->ball_alive[i] ? 1.0f : 0.0f;
        float x = env->ball_alive[i] ? env->ball_x[i] : 0.0f;
        float y = env->ball_alive[i] ? env->ball_y[i] : 0.0f;
        float vx = env->ball_alive[i] ? env->ball_vx[i] : 0.0f;
        float vy = env->ball_alive[i] ? env->ball_vy[i] : 0.0f;

        obs[idx++] = x * env->inv_table_width;
        obs[idx++] = y * env->inv_table_height;
        obs[idx++] = vx * env->inv_vel_scale;
        obs[idx++] = vy * env->inv_vel_scale;
        obs[idx++] = alive;
    }

    int opp = 1 - player;
    int my_group = env->player_group[player];
    int opp_group = env->player_group[opp];
    int my_remaining = my_group == GROUP_UNKNOWN ? 7 : count_remaining_group(env, my_group);
    int opp_remaining = opp_group == GROUP_UNKNOWN ? 7 : count_remaining_group(env, opp_group);

    obs[idx++] = env->current_player == player ? 1.0f : 0.0f;
    obs[idx++] = my_group == GROUP_SOLIDS ? 1.0f : 0.0f;
    obs[idx++] = my_group == GROUP_STRIPES ? 1.0f : 0.0f;
    obs[idx++] = opp_group == GROUP_SOLIDS ? 1.0f : 0.0f;
    obs[idx++] = opp_group == GROUP_STRIPES ? 1.0f : 0.0f;
    obs[idx++] = (float)my_remaining / 7.0f;
    obs[idx++] = (float)opp_remaining / 7.0f;
    obs[idx++] = env->ball_in_hand ? 1.0f : 0.0f;
    obs[idx++] = 1.0f - ((float)env->tick * env->inv_max_steps);
}

void compute_observations(Pool* env) {
    compute_single_observation(env, 0, env->observations + 0 * OBS_SIZE);
    compute_single_observation(env, 1, env->observations + 1 * OBS_SIZE);
}

static inline void add_episode_log(Pool* env) {
    float p0_score = 0.5f;
    if (env->winner == 0) p0_score = 1.0f;
    else if (env->winner == 1) p0_score = 0.0f;

    env->log.perf += p0_score;
    env->log.score += p0_score;
    env->log.episode_return += env->episode_return[0];
    env->log.episode_length += env->tick;
    env->log.shots += 0.5f * (env->shots_taken[0] + env->shots_taken[1]);
    env->log.n += 1.0f;
}

void init(Pool* env) {
    memset(&env->log, 0, sizeof(Log));
    env->client = NULL;
    update_derived_constants(env);
}

void c_reset(Pool* env) {
    env->tick = 0;
    env->winner = -1;
    env->current_player = rand() % NUM_PLAYERS;
    env->ball_in_hand = 0;

    for (int p = 0; p < NUM_PLAYERS; p++) {
        env->player_group[p] = GROUP_UNKNOWN;
        env->episode_return[p] = 0.0f;
        env->shots_taken[p] = 0;
        env->fouls[p] = 0;
    }

    rack_balls(env);
    compute_observations(env);
}

static inline void resolve_wall(Pool* env, int i) {
    if (!env->ball_alive[i]) return;

    float* x = &env->ball_x[i];
    float* y = &env->ball_y[i];
    float* vx = &env->ball_vx[i];
    float* vy = &env->ball_vy[i];
    float r = env->ball_radius;

    if (*x < r) {
        *x = r;
        *vx = fabsf(*vx) * env->restitution;
    } else if (*x > env->table_width - r) {
        *x = env->table_width - r;
        *vx = -fabsf(*vx) * env->restitution;
    }

    if (*y < r) {
        *y = r;
        *vy = fabsf(*vy) * env->restitution;
    } else if (*y > env->table_height - r) {
        *y = env->table_height - r;
        *vy = -fabsf(*vy) * env->restitution;
    }
}

static inline void collide_two_balls(Pool* env, int ia, int ib, int* first_hit) {
    if (!env->ball_alive[ia] || !env->ball_alive[ib]) return;

    float dx = env->ball_x[ib] - env->ball_x[ia];
    float dy = env->ball_y[ib] - env->ball_y[ia];
    float min_dist = 2.0f * env->ball_radius;
    float min_dist_sq = min_dist * min_dist;
    float d2 = dx * dx + dy * dy;

    if (d2 <= 1e-10f || d2 >= min_dist_sq) {
        return;
    }

    if (first_hit != NULL && *first_hit < 0 && (ia == CUE_BALL || ib == CUE_BALL)) {
        *first_hit = ia == CUE_BALL ? ib : ia;
    }

    float d = sqrtf(d2);
    float nx = dx / d;
    float ny = dy / d;

    float rel_vx = env->ball_vx[ib] - env->ball_vx[ia];
    float rel_vy = env->ball_vy[ib] - env->ball_vy[ia];
    float rel_normal = rel_vx * nx + rel_vy * ny;

    if (rel_normal < 0.0f) {
        float impulse = -(1.0f + env->restitution) * rel_normal * 0.5f;
        env->ball_vx[ia] -= impulse * nx;
        env->ball_vy[ia] -= impulse * ny;
        env->ball_vx[ib] += impulse * nx;
        env->ball_vy[ib] += impulse * ny;
    }

    float penetration = min_dist - d;
    float correction = 0.5f * penetration + 1e-4f;
    env->ball_x[ia] -= correction * nx;
    env->ball_y[ia] -= correction * ny;
    env->ball_x[ib] += correction * nx;
    env->ball_y[ib] += correction * ny;
}

static inline bool maybe_take_shot(Pool* env, bool balls_are_stationary) {
    int shooter = env->current_player;
    int action_base = shooter * 2;
    int direction_action = env->actions[action_base];
    int power_action = env->actions[action_base + 1];

    if (!balls_are_stationary) {
        return false;
    }

    if (env->ball_in_hand || !env->ball_alive[CUE_BALL]) {
        place_cue_ball_in_hand(env);
        env->ball_in_hand = 0;
    }

    int dir_idx = direction_action % NUM_SHOT_DIRS;
    if (dir_idx < 0) dir_idx += NUM_SHOT_DIRS;
    int pwr_idx = power_action % NUM_POWER_BINS;
    if (pwr_idx < 0) pwr_idx += NUM_POWER_BINS;
    float shot_impulse = env->shot_impulses[pwr_idx];

    env->ball_vx[CUE_BALL] += env->shot_cos[dir_idx] * shot_impulse;
    env->ball_vy[CUE_BALL] += env->shot_sin[dir_idx] * shot_impulse;

    env->shots_taken[shooter] += 1;
    env->rewards[shooter] += env->reward_shot;
    return true;
}

static inline bool simulate_physics_substep(Pool* env, unsigned char* potted, int* scratch, int* first_hit) {
    int alive_idx[NUM_BALLS];
    int moving_idx[NUM_BALLS];
    signed char moving_order[NUM_BALLS];
    unsigned char moving_mask[NUM_BALLS] = {0};
    int alive_count = 0;
    int moving_count = 0;

    for (int i = 0; i < NUM_BALLS; i++) {
        moving_order[i] = -1;
        if (!env->ball_alive[i]) continue;
        alive_idx[alive_count++] = i;
        if (env->ball_vx[i] != 0.0f || env->ball_vy[i] != 0.0f) {
            moving_order[i] = (signed char)moving_count;
            moving_mask[i] = 1;
            moving_idx[moving_count++] = i;
            env->ball_x[i] += env->ball_vx[i];
            env->ball_y[i] += env->ball_vy[i];
        }
    }

    for (int k = 0; k < moving_count; k++) {
        resolve_wall(env, moving_idx[k]);
    }

    // Only evaluate pairs touched by currently-moving balls.
    // If a stationary ball starts moving due to collision, enqueue it once.
    for (int m = 0; m < moving_count; m++) {
        int ia = moving_idx[m];
        int ia_order = moving_order[ia];
        for (int b = 0; b < alive_count; b++) {
            int ib = alive_idx[b];
            if (ib == ia) {
                continue;
            }

            if (moving_mask[ib] && moving_order[ib] < ia_order) {
                continue;
            }

            collide_two_balls(env, ia, ib, first_hit);

            if (!moving_mask[ib] && (env->ball_vx[ib] != 0.0f || env->ball_vy[ib] != 0.0f)) {
                moving_order[ib] = (signed char)moving_count;
                moving_mask[ib] = 1;
                moving_idx[moving_count++] = ib;
            }
        }
    }

    for (int k = 0; k < moving_count; k++) {
        int i = moving_idx[k];
        if (!env->ball_alive[i]) continue;

        env->ball_vx[i] *= env->friction;
        env->ball_vy[i] *= env->friction;

        if (fabsf(env->ball_vx[i]) < STATIONARY_EPS) env->ball_vx[i] = 0.0f;
        if (fabsf(env->ball_vy[i]) < STATIONARY_EPS) env->ball_vy[i] = 0.0f;
    }

    bool moving = false;
    for (int k = 0; k < alive_count; k++) {
        int i = alive_idx[k];
        if (!env->ball_alive[i]) continue;

        for (int p = 0; p < NUM_POCKETS; p++) {
            if (dist_sq(env->ball_x[i], env->ball_y[i], env->pocket_x[p], env->pocket_y[p]) <= env->pocket_radius_sq) {
                env->ball_alive[i] = 0;
                env->ball_vx[i] = 0.0f;
                env->ball_vy[i] = 0.0f;
                potted[i] = 1;
                if (i == CUE_BALL) {
                    *scratch = 1;
                }
                break;
            }
        }

        if (env->ball_alive[i] && (env->ball_vx[i] != 0.0f || env->ball_vy[i] != 0.0f)) {
            moving = true;
        }
    }

    return moving;
}

static inline void evaluate_shot(Pool* env, int shooter, unsigned char* potted, int scratch, int first_hit, bool* done) {
    int opp = 1 - shooter;

    int solids_potted = 0;
    int stripes_potted = 0;
    for (int b = 1; b <= 7; b++) solids_potted += potted[b] ? 1 : 0;
    for (int b = 9; b <= 15; b++) stripes_potted += potted[b] ? 1 : 0;

    bool eight_potted = potted[EIGHT_BALL] != 0;
    bool cue_potted = potted[CUE_BALL] != 0 || scratch;

    int shooter_group = env->player_group[shooter];
    bool foul = false;
    if (cue_potted) foul = true;
    if (first_hit < 0) foul = true;
    if (shooter_group == GROUP_UNKNOWN && first_hit == EIGHT_BALL) foul = true;

    int own_remaining = shooter_group == GROUP_UNKNOWN ? 7 : count_remaining_group(env, shooter_group);

    if (shooter_group != GROUP_UNKNOWN && first_hit >= 0) {
        if (own_remaining > 0) {
            if (group_for_ball(first_hit) != shooter_group) {
                foul = true;
            }
        } else {
            if (first_hit != EIGHT_BALL) {
                foul = true;
            }
        }
    }

    if (shooter_group == GROUP_UNKNOWN && !foul && !cue_potted && !eight_potted) {
        if (solids_potted > 0 && stripes_potted == 0) {
            env->player_group[shooter] = GROUP_SOLIDS;
            env->player_group[opp] = GROUP_STRIPES;
        } else if (stripes_potted > 0 && solids_potted == 0) {
            env->player_group[shooter] = GROUP_STRIPES;
            env->player_group[opp] = GROUP_SOLIDS;
        }
        shooter_group = env->player_group[shooter];
        own_remaining = shooter_group == GROUP_UNKNOWN ? 7 : count_remaining_group(env, shooter_group);
    }

    int own_potted = 0;
    int opp_potted = 0;
    if (shooter_group == GROUP_SOLIDS) {
        own_potted = solids_potted;
        opp_potted = stripes_potted;
    } else if (shooter_group == GROUP_STRIPES) {
        own_potted = stripes_potted;
        opp_potted = solids_potted;
    } else {
        own_potted = solids_potted + stripes_potted;
        opp_potted = 0;
    }

    if (own_potted > 0) {
        env->rewards[shooter] += env->reward_legal_pot * own_potted;
    }
    if (opp_potted > 0) {
        env->rewards[shooter] += env->reward_illegal_pot * opp_potted;
    }

    if (eight_potted) {
        if (shooter_group != GROUP_UNKNOWN && own_remaining == 0 && !foul && !cue_potted) {
            env->winner = shooter;
        } else {
            env->winner = opp;
        }
        *done = true;
        return;
    }

    if (foul) {
        env->fouls[shooter] += 1;
        env->rewards[shooter] += env->reward_foul;
        env->current_player = opp;
        env->ball_in_hand = 1;
        return;
    }

    bool keep_turn = false;
    if (shooter_group == GROUP_UNKNOWN) {
        keep_turn = (solids_potted + stripes_potted) > 0;
    } else {
        keep_turn = own_potted > 0;
    }

    if (!keep_turn) {
        env->current_player = opp;
    }
}

void c_step(Pool* env) {
    for (int p = 0; p < NUM_PLAYERS; p++) {
        env->rewards[p] = 0.0f;
        env->terminals[p] = 0;
        env->truncations[p] = 0;
    }

    env->tick += 1;
    int shooter = env->current_player;

    bool stationary = balls_stationary(env);
    bool shot_fired = maybe_take_shot(env, stationary);
    bool done = false;
    bool timeout = false;

    if (shot_fired || !stationary) {
        unsigned char potted[NUM_BALLS];
        memset(potted, 0, sizeof(potted));
        int scratch = 0;
        int first_hit = -1;

        int max_substeps = env->fast_forward ? env->max_physics_steps : 1;
        if (max_substeps < 1) max_substeps = 1;

        bool moving = true;
        for (int s = 0; s < max_substeps; s++) {
            moving = simulate_physics_substep(env, potted, &scratch, &first_hit);
            env->rewards[shooter] += env->reward_step;

            if (env->fast_forward && !moving) {
                break;
            }
        }

        if (shot_fired) {
            evaluate_shot(env, shooter, potted, scratch, first_hit, &done);
        }
    } else {
        env->rewards[shooter] -= 0.01f;
    }

    if (!done && env->winner >= 0) {
        done = true;
    }

    if (!done && env->tick >= env->max_steps) {
        done = true;
        timeout = true;
        env->winner = -1;
    }

    if (done && env->winner >= 0) {
        int loser = 1 - env->winner;
        env->rewards[env->winner] += env->reward_win;
        env->rewards[loser] -= env->reward_win;
    }

    for (int p = 0; p < NUM_PLAYERS; p++) {
        env->episode_return[p] += env->rewards[p];
    }

    if (done) {
        env->terminals[0] = 1;
        env->terminals[1] = 1;
        if (timeout) {
            env->truncations[0] = 1;
            env->truncations[1] = 1;
        }

        add_episode_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

static Client* make_client(Pool* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->margin = 40;
    client->scale = 280.0f;
    client->width_px = (int)(env->table_width * client->scale) + 2 * client->margin;
    client->height_px = (int)(env->table_height * client->scale) + 2 * client->margin;

    InitWindow(client->width_px, client->height_px, "PufferLib Pool 8-ball Lite");
    SetTargetFPS(60);
    return client;
}

static Color ball_color(int ball) {
    if (ball == CUE_BALL) return RAYWHITE;
    if (ball == EIGHT_BALL) return BLACK;
    if (group_for_ball(ball) == GROUP_SOLIDS) return (Color){220, 70, 40, 255};
    return (Color){55, 120, 230, 255};
}

void c_render(Pool* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = make_client(env);
    }

    Client* client = env->client;
    int margin = client->margin;
    int table_w = (int)(env->table_width * client->scale);
    int table_h = (int)(env->table_height * client->scale);

    int pocket_r = (int)(env->pocket_radius * client->scale);
    int ball_r = (int)(env->ball_radius * client->scale);

    BeginDrawing();
    ClearBackground((Color){18, 27, 38, 255});

    DrawRectangle(margin - 10, margin - 10, table_w + 20, table_h + 20, (Color){94, 61, 34, 255});
    DrawRectangle(margin, margin, table_w, table_h, (Color){30, 122, 68, 255});

    for (int p = 0; p < NUM_POCKETS; p++) {
        int px = margin + (int)(env->pocket_x[p] * client->scale);
        int py = margin + (int)(env->pocket_y[p] * client->scale);
        DrawCircle(px, py, pocket_r, BLACK);
    }

    for (int i = 0; i < NUM_BALLS; i++) {
        if (!env->ball_alive[i]) continue;

        int x = margin + (int)(env->ball_x[i] * client->scale);
        int y = margin + (int)(env->ball_y[i] * client->scale);
        Color c = ball_color(i);

        if (i == CUE_BALL || i == EIGHT_BALL || group_for_ball(i) == GROUP_SOLIDS) {
            DrawCircle(x, y, ball_r, c);
            DrawCircleLines(x, y, ball_r, LIGHTGRAY);
        } else {
            DrawCircle(x, y, ball_r, RAYWHITE);
            DrawCircle(x, y, (int)(ball_r * 0.62f), c);
            DrawCircleLines(x, y, ball_r, LIGHTGRAY);
        }
    }

    int p0 = env->player_group[0];
    int p1 = env->player_group[1];
    DrawText(TextFormat("Step: %d / %d", env->tick, env->max_steps), 12, 10, 20, RAYWHITE);
    DrawText(TextFormat("Turn: P%d", env->current_player + 1), 12, 34, 20, RAYWHITE);
    DrawText(TextFormat("P1: %s  shots=%d  fouls=%d", group_name(p0), env->shots_taken[0], env->fouls[0]), 12, 58, 18, LIGHTGRAY);
    DrawText(TextFormat("P2: %s  shots=%d  fouls=%d", group_name(p1), env->shots_taken[1], env->fouls[1]), 12, 80, 18, LIGHTGRAY);
    DrawText("Shift+arrows to shoot, 1..5 power", 12, 104, 18, LIGHTGRAY);

    EndDrawing();
}

void c_close(Pool* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}
