#include <stdlib.h>
#include "pool.h"

int main() {
    Pool env = {
        .table_width = 2.84f,
        .table_height = 1.42f,
        .ball_radius = 0.03f,
        .pocket_radius = 0.06f,
        .friction = 0.992f,
        .restitution = 0.96f,
        .impulse = 0.12f,
        .reward_step = -0.001f,
        .reward_pot_object = 1.0f,
        .reward_scratch = -0.5f,
        .max_steps = 300,
    };

    env.observations = (float*)calloc(11, sizeof(float));
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));

    init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0;

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyPressed(KEY_RIGHT)) env.actions[0] = 1;   // 0 deg
            if (IsKeyPressed(KEY_DOWN)) env.actions[0] = 5;    // 90 deg
            if (IsKeyPressed(KEY_LEFT)) env.actions[0] = 9;    // 180 deg
            if (IsKeyPressed(KEY_UP)) env.actions[0] = 13;     // 270 deg
        } else {
            if ((rand() % 18) == 0) {
                env.actions[0] = 1 + (rand() % 16);
            }
        }

        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);

    return 0;
}
