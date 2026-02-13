#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "raylib.h"

#define PI 3.14159265358979323846f
#define NUM_SHOT_DIRS 16
#define STATIONARY_EPS 0.0005f

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

    float reward_step;
    float reward_pot_object;
    float reward_scratch;

    int max_steps;

    float cue_x;
    float cue_y;
    float cue_vx;
    float cue_vy;

    float obj_x;
    float obj_y;
    float obj_vx;
    float obj_vy;

    float pocket_x;
    float pocket_y;

    int tick;
    float episode_return;
    int shots_taken;
};

static inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float randf(float lo, float hi) {
    return lo + ((float)rand() / (float)RAND_MAX) * (hi - lo);
}

static inline float dist_sq(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return dx * dx + dy * dy;
}

static inline bool balls_stationary(Pool* env) {
    float cue_speed_sq = env->cue_vx * env->cue_vx + env->cue_vy * env->cue_vy;
    float obj_speed_sq = env->obj_vx * env->obj_vx + env->obj_vy * env->obj_vy;
    return cue_speed_sq < STATIONARY_EPS * STATIONARY_EPS && obj_speed_sq < STATIONARY_EPS * STATIONARY_EPS;
}

static inline void choose_random_pocket(Pool* env) {
    float px = env->pocket_radius * 1.2f;
    float py = env->pocket_radius * 1.2f;

    switch (rand() % 4) {
        case 0:
            env->pocket_x = px;
            env->pocket_y = py;
            break;
        case 1:
            env->pocket_x = env->table_width - px;
            env->pocket_y = py;
            break;
        case 2:
            env->pocket_x = px;
            env->pocket_y = env->table_height - py;
            break;
        default:
            env->pocket_x = env->table_width - px;
            env->pocket_y = env->table_height - py;
            break;
    }
}

static inline void sample_balls(Pool* env) {
    float margin = env->ball_radius * 2.0f;
    float min_dist = env->ball_radius * 5.0f;
    float min_dist_sq = min_dist * min_dist;

    for (int i = 0; i < 1024; i++) {
        env->cue_x = randf(margin, env->table_width - margin);
        env->cue_y = randf(margin, env->table_height - margin);
        env->obj_x = randf(margin, env->table_width - margin);
        env->obj_y = randf(margin, env->table_height - margin);

        if (dist_sq(env->cue_x, env->cue_y, env->obj_x, env->obj_y) < min_dist_sq) {
            continue;
        }

        float pocket_zone = (env->pocket_radius + env->ball_radius) * (env->pocket_radius + env->ball_radius);
        if (dist_sq(env->cue_x, env->cue_y, env->pocket_x, env->pocket_y) < pocket_zone) {
            continue;
        }
        if (dist_sq(env->obj_x, env->obj_y, env->pocket_x, env->pocket_y) < pocket_zone) {
            continue;
        }

        return;
    }

    env->cue_x = env->table_width * 0.25f;
    env->cue_y = env->table_height * 0.50f;
    env->obj_x = env->table_width * 0.60f;
    env->obj_y = env->table_height * 0.50f;
}

void compute_observations(Pool* env) {
    float vel_scale = fmaxf(env->impulse, 1e-3f);

    env->observations[0] = env->cue_x / env->table_width;
    env->observations[1] = env->cue_y / env->table_height;
    env->observations[2] = env->cue_vx / vel_scale;
    env->observations[3] = env->cue_vy / vel_scale;

    env->observations[4] = env->obj_x / env->table_width;
    env->observations[5] = env->obj_y / env->table_height;
    env->observations[6] = env->obj_vx / vel_scale;
    env->observations[7] = env->obj_vy / vel_scale;

    env->observations[8] = env->pocket_x / env->table_width;
    env->observations[9] = env->pocket_y / env->table_height;

    env->observations[10] = 1.0f - ((float)env->tick / (float)env->max_steps);
}

static inline void add_episode_log(Pool* env, bool object_potted) {
    env->log.perf += object_potted ? 1.0f : 0.0f;
    env->log.score += object_potted ? 1.0f : 0.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.shots += env->shots_taken;
    env->log.n += 1.0f;
}

void init(Pool* env) {
    memset(&env->log, 0, sizeof(Log));
    env->client = NULL;
}

void c_reset(Pool* env) {
    env->tick = 0;
    env->episode_return = 0.0f;
    env->shots_taken = 0;

    choose_random_pocket(env);
    sample_balls(env);

    env->cue_vx = 0.0f;
    env->cue_vy = 0.0f;
    env->obj_vx = 0.0f;
    env->obj_vy = 0.0f;

    compute_observations(env);
}

static inline void resolve_wall(float* x, float* y, float* vx, float* vy, float radius, float width, float height, float restitution) {
    if (*x < radius) {
        *x = radius;
        *vx = fabsf(*vx) * restitution;
    } else if (*x > width - radius) {
        *x = width - radius;
        *vx = -fabsf(*vx) * restitution;
    }

    if (*y < radius) {
        *y = radius;
        *vy = fabsf(*vy) * restitution;
    } else if (*y > height - radius) {
        *y = height - radius;
        *vy = -fabsf(*vy) * restitution;
    }
}

static inline void collide_balls(Pool* env) {
    float dx = env->obj_x - env->cue_x;
    float dy = env->obj_y - env->cue_y;
    float min_dist = 2.0f * env->ball_radius;
    float min_dist_sq = min_dist * min_dist;
    float d2 = dx * dx + dy * dy;

    if (d2 <= 1e-10f || d2 >= min_dist_sq) {
        return;
    }

    float d = sqrtf(d2);
    float nx = dx / d;
    float ny = dy / d;

    float rel_vx = env->obj_vx - env->cue_vx;
    float rel_vy = env->obj_vy - env->cue_vy;
    float rel_normal = rel_vx * nx + rel_vy * ny;

    if (rel_normal < 0.0f) {
        float impulse = -(1.0f + env->restitution) * rel_normal * 0.5f;
        env->cue_vx -= impulse * nx;
        env->cue_vy -= impulse * ny;
        env->obj_vx += impulse * nx;
        env->obj_vy += impulse * ny;
    }

    float penetration = min_dist - d;
    float correction = 0.5f * penetration + 1e-4f;
    env->cue_x -= correction * nx;
    env->cue_y -= correction * ny;
    env->obj_x += correction * nx;
    env->obj_y += correction * ny;
}

static inline void maybe_take_shot(Pool* env) {
    int action = env->actions[0];
    if (action <= 0 || !balls_stationary(env)) {
        return;
    }

    int dir_idx = (action - 1) % NUM_SHOT_DIRS;
    float angle = (2.0f * PI * (float)dir_idx) / (float)NUM_SHOT_DIRS;
    env->cue_vx += cosf(angle) * env->impulse;
    env->cue_vy += sinf(angle) * env->impulse;
    env->shots_taken += 1;
}

void c_step(Pool* env) {
    env->rewards[0] = env->reward_step;
    env->terminals[0] = 0;

    env->tick += 1;
    maybe_take_shot(env);

    env->cue_x += env->cue_vx;
    env->cue_y += env->cue_vy;
    env->obj_x += env->obj_vx;
    env->obj_y += env->obj_vy;

    resolve_wall(&env->cue_x, &env->cue_y, &env->cue_vx, &env->cue_vy,
        env->ball_radius, env->table_width, env->table_height, env->restitution);
    resolve_wall(&env->obj_x, &env->obj_y, &env->obj_vx, &env->obj_vy,
        env->ball_radius, env->table_width, env->table_height, env->restitution);

    collide_balls(env);

    env->cue_vx *= env->friction;
    env->cue_vy *= env->friction;
    env->obj_vx *= env->friction;
    env->obj_vy *= env->friction;

    if (fabsf(env->cue_vx) < STATIONARY_EPS) env->cue_vx = 0.0f;
    if (fabsf(env->cue_vy) < STATIONARY_EPS) env->cue_vy = 0.0f;
    if (fabsf(env->obj_vx) < STATIONARY_EPS) env->obj_vx = 0.0f;
    if (fabsf(env->obj_vy) < STATIONARY_EPS) env->obj_vy = 0.0f;

    bool object_potted = dist_sq(env->obj_x, env->obj_y, env->pocket_x, env->pocket_y) <= env->pocket_radius * env->pocket_radius;
    bool cue_potted = dist_sq(env->cue_x, env->cue_y, env->pocket_x, env->pocket_y) <= env->pocket_radius * env->pocket_radius;

    bool done = false;
    bool success = false;

    if (object_potted) {
        env->rewards[0] += env->reward_pot_object;
        done = true;
        success = true;
    }

    if (cue_potted) {
        env->rewards[0] += env->reward_scratch;
        done = true;
    }

    if (env->tick >= env->max_steps) {
        done = true;
    }

    env->episode_return += env->rewards[0];

    if (done) {
        env->terminals[0] = 1;
        add_episode_log(env, success);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

static Client* make_client(Pool* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->margin = 40;
    client->scale = 320.0f;
    client->width_px = (int)(env->table_width * client->scale) + 2 * client->margin;
    client->height_px = (int)(env->table_height * client->scale) + 2 * client->margin;

    InitWindow(client->width_px, client->height_px, "PufferLib Pool");
    SetTargetFPS(60);
    return client;
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

    int pocket_x = margin + (int)(env->pocket_x * client->scale);
    int pocket_y = margin + (int)(env->pocket_y * client->scale);
    int cue_x = margin + (int)(env->cue_x * client->scale);
    int cue_y = margin + (int)(env->cue_y * client->scale);
    int obj_x = margin + (int)(env->obj_x * client->scale);
    int obj_y = margin + (int)(env->obj_y * client->scale);

    int pocket_r = (int)(env->pocket_radius * client->scale);
    int ball_r = (int)(env->ball_radius * client->scale);

    BeginDrawing();
    ClearBackground((Color){18, 27, 38, 255});

    DrawRectangle(margin - 10, margin - 10, table_w + 20, table_h + 20, (Color){94, 61, 34, 255});
    DrawRectangle(margin, margin, table_w, table_h, (Color){25, 115, 62, 255});

    DrawCircle(pocket_x, pocket_y, pocket_r, BLACK);

    DrawCircle(cue_x, cue_y, ball_r, RAYWHITE);
    DrawCircleLines(cue_x, cue_y, ball_r, LIGHTGRAY);

    DrawCircle(obj_x, obj_y, ball_r, (Color){200, 40, 40, 255});
    DrawCircleLines(obj_x, obj_y, ball_r, (Color){255, 200, 200, 255});

    DrawText(TextFormat("Step: %d / %d", env->tick, env->max_steps), 12, 10, 20, RAYWHITE);
    DrawText(TextFormat("Shots: %d", env->shots_taken), 12, 34, 20, RAYWHITE);
    DrawText("Shift + arrows in pool.c demo for manual shots", 12, 58, 18, LIGHTGRAY);

    EndDrawing();
}

void c_close(Pool* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}
