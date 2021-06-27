from gym.envs.registration import register

register(
    id='ball-v0',
    entry_point='gym_ball.envs:BallEnv',
)