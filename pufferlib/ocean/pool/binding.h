#define POOL_STATIC_ACTION_INT 1
#define STATIC_BINDING
#include "pool.h"

#define OBS_SIZE 89
#define NUM_ATNS 2
#define ACT_SIZES {16, 5}
#define OBS_TYPE FLOAT
#define ACT_TYPE INT

#define Env Pool
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 2;

    env->table_width = dict_get(kwargs, "width")->value;
    env->table_height = dict_get(kwargs, "height")->value;
    env->ball_radius = dict_get(kwargs, "ball_radius")->value;
    env->pocket_radius = dict_get(kwargs, "pocket_radius")->value;
    env->friction = dict_get(kwargs, "friction")->value;
    env->restitution = dict_get(kwargs, "restitution")->value;
    env->impulse = dict_get(kwargs, "impulse")->value;
    env->min_power = dict_get(kwargs, "min_power")->value;

    env->reward_step = dict_get(kwargs, "reward_step")->value;
    env->reward_shot = dict_get(kwargs, "reward_shot")->value;
    env->reward_legal_pot = dict_get(kwargs, "reward_legal_pot")->value;
    env->reward_illegal_pot = dict_get(kwargs, "reward_illegal_pot")->value;
    env->reward_foul = dict_get(kwargs, "reward_foul")->value;
    env->reward_win = dict_get(kwargs, "reward_win")->value;

    env->fast_forward = (unsigned char)dict_get(kwargs, "fast_forward")->value;
    env->max_physics_steps = (int)dict_get(kwargs, "max_physics_steps")->value;
    env->max_steps = (int)dict_get(kwargs, "max_steps")->value;

    if (env->table_width <= 0.0f) env->table_width = 2.84f;
    if (env->table_height <= 0.0f) env->table_height = 1.42f;
    if (env->ball_radius <= 0.0f) env->ball_radius = 0.03f;
    if (env->pocket_radius <= 0.0f) env->pocket_radius = 0.06f;

    if (env->friction <= 0.0f || env->friction > 1.0f) env->friction = 0.992f;
    if (env->restitution < 0.0f || env->restitution > 1.0f) env->restitution = 0.96f;
    if (env->impulse <= 0.0f) env->impulse = 0.12f;
    if (env->min_power < 0.0f || env->min_power > 1.0f) env->min_power = 0.35f;

    if (env->max_physics_steps < 1) env->max_physics_steps = 96;
    env->fast_forward = env->fast_forward ? 1 : 0;
    if (env->max_steps < 1) env->max_steps = 200;

    // Static runtime uses a terminals-only path by default.
    env->truncations = NULL;

    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "shots", log->shots);
}
