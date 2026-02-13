#include "pool.h"

#define Env Pool
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->table_width = unpack(kwargs, "width");
    env->table_height = unpack(kwargs, "height");
    env->ball_radius = unpack(kwargs, "ball_radius");
    env->pocket_radius = unpack(kwargs, "pocket_radius");
    env->friction = unpack(kwargs, "friction");
    env->restitution = unpack(kwargs, "restitution");
    env->impulse = unpack(kwargs, "impulse");
    env->reward_step = unpack(kwargs, "reward_step");
    env->reward_pot_object = unpack(kwargs, "reward_pot_object");
    env->reward_scratch = unpack(kwargs, "reward_scratch");
    env->max_steps = unpack(kwargs, "max_steps");

    if (env->table_width <= 0.0f) env->table_width = 2.84f;
    if (env->table_height <= 0.0f) env->table_height = 1.42f;
    if (env->ball_radius <= 0.0f) env->ball_radius = 0.03f;
    if (env->pocket_radius <= 0.0f) env->pocket_radius = 0.06f;

    if (env->friction <= 0.0f || env->friction > 1.0f) env->friction = 0.992f;
    if (env->restitution < 0.0f || env->restitution > 1.0f) env->restitution = 0.96f;
    if (env->impulse <= 0.0f) env->impulse = 0.12f;
    if (env->max_steps < 1) env->max_steps = 300;

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "shots", log->shots);
    return 0;
}
