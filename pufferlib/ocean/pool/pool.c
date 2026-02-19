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
        .min_power = 0.35f,
        .reward_step = 0.0f,
        .reward_shot = -0.001f,
        .reward_legal_pot = 0.03f,
        .reward_illegal_pot = -0.015f,
        .reward_foul = -0.05f,
        .reward_win = 1.0f,
        .fast_forward = 0,
        .max_physics_steps = 96,
        .max_steps = 200,
    };

    env.observations = (float*)calloc(NUM_PLAYERS * OBS_SIZE, sizeof(float));
    env.actions = (int*)calloc(NUM_PLAYERS * 2, sizeof(int));
    env.rewards = (float*)calloc(NUM_PLAYERS, sizeof(float));
    env.terminals = (unsigned char*)calloc(NUM_PLAYERS, sizeof(unsigned char));
    env.truncations = (unsigned char*)calloc(NUM_PLAYERS, sizeof(unsigned char));

    init(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 0;
        env.actions[1] = 4;
        env.actions[2] = 0;
        env.actions[3] = 4;

        if (env.current_player == 0) {
            if (IsKeyDown(KEY_LEFT_SHIFT)) {
                if (IsKeyPressed(KEY_RIGHT)) env.actions[0] = 0;
                if (IsKeyPressed(KEY_DOWN)) env.actions[0] = 4;
                if (IsKeyPressed(KEY_LEFT)) env.actions[0] = 8;
                if (IsKeyPressed(KEY_UP)) env.actions[0] = 12;
                if (IsKeyPressed(KEY_ONE)) env.actions[1] = 0;
                if (IsKeyPressed(KEY_TWO)) env.actions[1] = 1;
                if (IsKeyPressed(KEY_THREE)) env.actions[1] = 2;
                if (IsKeyPressed(KEY_FOUR)) env.actions[1] = 3;
                if (IsKeyPressed(KEY_FIVE)) env.actions[1] = 4;
            }
        } else {
            if ((rand() % 18) == 0) {
                env.actions[2] = rand() % 16;
                env.actions[3] = rand() % 5;
            }
        }

        c_step(&env);
        c_render(&env);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.truncations);
    c_close(&env);

    return 0;
}
