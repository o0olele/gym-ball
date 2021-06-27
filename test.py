import gym

env = gym.make('gym_ball:ball-v0')

while True:
    env.step(150)
    env.render()