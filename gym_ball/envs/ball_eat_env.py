import math
import time

import gym
from gym import spaces
from gym.envs.classic_control import rendering

import numpy as np

VIEWPORT_W = 200
VIEWPORT_H = 200

AGENT_INIT_SCORE = 5
MAX_FRAME = 5000
MAX_FOOD_NUM = 20
MAX_FOOD_SCORE = 10
MAX_BALL_SCORE = 200

def CheckBound(low, high, value):
    if value > high:
        value -= (high - low)
    elif value < low:
        value += (high - low)
    value = np.clip(value, low, high)
    return value

class Ball():
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s
        self.w = 0

        self.lastupdate = 0   # last update time, used to caculate ball move
        self.timescale = 1  # time scale, used to caculate ball move

    def radius(self):
        return math.sqrt(self.s / math.pi)

    def state(self):
        return [self.x, self.y, self.s]

    def addscore(self, s):
        self.s += s
        self.s = np.clip(self.s, 0, MAX_BALL_SCORE)

    def update(self, way, frame):
        raise NotImplementedError

class Agent(Ball):
    def update(self, way, frame):
        speed = 5.0 / self.s  # score to speed
        now = frame  # now time

        self.w = way * 2 * math.pi / 360.0  # angle to radius

        self.x += math.cos(self.w) * speed * (now - self.lastupdate) * self.timescale  # speed * time = distance
        self.y += math.sin(self.w) * speed * (now - self.lastupdate) * self.timescale  # speed * time = distance

        self.x = CheckBound(0, VIEWPORT_W, self.x)
        self.y = CheckBound(0, VIEWPORT_H, self.y)

        self.lastupdate = now  # update time

class Food(Ball):
    def update(self, way, frame):
        pass

class BallEatEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.viewer = None  # render viewer
        self.scale = 2      # render scale
        self.frame = 0

        '''
            0~35
        '''
        self.action_space = spaces.Discrete(36)
        '''
            [[x,y,score], agent ball
            [x,y,score], food ball
            ...]
        '''
        self.observation_space = spaces.Box(low=0, high=MAX_BALL_SCORE, shape=((MAX_FOOD_NUM + 1) * 3, ))

        self.agent = None
        self.foods = []

        self.reset()

    def reset(self):
        self.foods = []
        self.frame = 0

        self.agent = self.rand_agent()

        for i in range(MAX_FOOD_NUM):
            self.foods.append(self.rand_food())

        self.state = np.concatenate(([self.agent.state()], [food.state() for (_, food) in enumerate(self.foods)]))

        return self.state.reshape((MAX_FOOD_NUM + 1) * 3, )

    def step(self, action: int):
        reward = 0.0
        done = False

        action = action * 10

        self.frame += 1
        self.agent.update(action, self.frame)

        radius = self.agent.radius()
        for _, food in enumerate(self.foods):

            dis_x = math.fabs(self.agent.x - food.x)
            dis_y = math.fabs(self.agent.y - food.y)

            if self.agent.s <= food.s:
                continue

            if dis_x > radius or dis_y > radius:
                continue

            if dis_x*dis_x + dis_y*dis_y > radius*radius:
                continue

            reward += food.s

            self.agent.addscore(food.s)
            self.foods.remove(food)

        for _ in range(MAX_FOOD_NUM - len(self.foods)):
            self.foods.append(self.rand_food())

        if self.agent.s >= MAX_BALL_SCORE or self.frame >= MAX_FRAME:
            done = True

        if done:
            reward += (MAX_FRAME - self.frame) * 0.05 + (self.agent.s - MAX_BALL_SCORE)

        self.state = np.concatenate(([self.agent.state()], [food.state() for (_, food) in enumerate(self.foods)]))

        return self.state.reshape((MAX_FOOD_NUM + 1) * 3, ), reward, done, {}

    def render(self, mode='human'):
        # create viewer
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W * self.scale, VIEWPORT_H * self.scale)

        self.draw_circle(self.agent.x, self.agent.y, self.agent.radius(), (0, 0, 255))

        for _, food in enumerate(self.foods):
            self.draw_circle(food.x, food.y, food.radius(), (255, 0, 0))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        return

    def draw_circle(self, x, y, r, c):
        assert isinstance(self.viewer, rendering.Viewer)

        transform = rendering.Transform()
        transform.set_translation(x * self.scale, y * self.scale)
        self.viewer.draw_circle(r * self.scale, 30, color=c).add_attr(transform)

    @staticmethod
    def rand_food():
        return Food(np.random.rand(1)[0]*VIEWPORT_W, np.random.rand(1)[0]*VIEWPORT_H, np.random.rand(1)[0] * MAX_FOOD_SCORE)

    @staticmethod
    def rand_agent():
        return Agent(np.random.rand(1)[0]*VIEWPORT_W, np.random.rand(1)[0]*VIEWPORT_H, AGENT_INIT_SCORE)

if __name__ == '__main__':
    env = BallEatEnv()

    while True:
        env.step(15)
        env.render()
