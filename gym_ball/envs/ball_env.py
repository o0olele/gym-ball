import math
import time

import gym
from gym import spaces

import numpy as np

VIEWPORT_W = 200
VIEWPORT_H = 200

MAX_BALL_NUM = 10
MAX_BALL_SCORE = 20

BALL_TYPE_OTHER = 0
BALL_TYPE_SELF = 1

BALL_START_ID = 0

def GenerateBallID():
    global BALL_START_ID

    BALL_START_ID += 1

    return BALL_START_ID

def CheckBound(low, high, value):
    if value > high:
        value -= (high - low)
    elif value < low:
        value += (high - low)
    return value

class Ball():
    def __init__(self, x: np.float32, y: np.float32, score: np.float32, way: np.int, t: np.int):
        '''
            x   coordinate
            y   coordinate
            s   score of ball
            w   move direction of ball, in radius
            t   type of ball, self or other
        '''
        self.x = x
        self.y = y
        self.s = score
        self.w = way * 2 * math.pi / 360.0  # angle to radius
        self.t = t

        self.id = GenerateBallID()      # ball id
        self.lastupdate = time.time()   # last update time, used to caculate ball move
        self.timescale = 100            # time scale, used to caculate ball move

    def update(self, way):
        '''
            update ball, include position
        '''

        # can only change self way
        if self.t == BALL_TYPE_SELF:
            self.w = way * 2 * math.pi / 360.0  # angle to radius

        speed = 1.0 / self.s    # score to speed
        now = time.time()       # now time

        self.x += math.cos(self.w) * speed * (now - self.lastupdate) * self.timescale   # speed * time = distance
        self.y += math.sin(self.w) * speed * (now - self.lastupdate) * self.timescale   # speed * time = distance

        self.x = CheckBound(0, VIEWPORT_W, self.x)
        self.y = CheckBound(0, VIEWPORT_H, self.y)

        self.lastupdate = now   # update time

    def addscore(self, score: np.float32):
        self.s += score

    def state(self):
        return [self.x, self.y, self.s, self.t]


class BallEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.viewer = None  # render viewer
        self.scale = 5      # render scale

        '''
            0~359
        '''
        self.action_space = spaces.Discrete(360)
        ''' [[x, y, score, type],
             [x, y, score, type],
             ...                ]
        '''
        self.observation_space = spaces.Box(low=0, high=VIEWPORT_H, shape=(MAX_BALL_NUM, 4), dtype=np.float32)

        self.balls = []

        self.state = np.zeros(shape=(MAX_BALL_NUM, 4), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balls = []

        # random gen other balls
        for i in range(MAX_BALL_NUM - 1):
            self.balls.append(self.randball(BALL_TYPE_OTHER))

        # random gen self ball
        self.selfball = self.randball(BALL_TYPE_SELF)

        # add to ball list
        self.balls.append(self.selfball)

        # update state
        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])

        return self.state

    def step(self, action):
        reward = 0.0
        done = False

        # update ball
        for _, ball in enumerate(self.balls):
            ball.update(action)

        '''
            Calculate Ball Eat
            if ball A contains ball B's center, and A's score > B's score, A eats B
        '''
        _new_ball_types = []
        for _, A_ball in enumerate(self.balls):
            for _, B_ball in enumerate(self.balls):

                if A_ball.id == B_ball.id:
                    continue

                # radius of ball A
                A_radius = math.sqrt(A_ball.s / math.pi)

                # vector AB
                AB_x = math.fabs(A_ball.x - B_ball.x)
                AB_y = math.fabs(A_ball.y - B_ball.y)

                # B is out of A
                if AB_x > A_radius or AB_y > A_radius:
                    continue

                # B is out of A
                if AB_x*AB_x + AB_y*AB_y > A_radius*A_radius:
                    continue

                # if self ball be eaten
                if B_ball.t == BALL_TYPE_SELF:
                    done = True

                # A eat B
                A_ball.addscore(B_ball.s)

                # Caculate reward
                if A_ball.t == BALL_TYPE_SELF:
                    reward += B_ball.s

                # delete B
                _new_ball_types.append(B_ball.t)
                self.balls.remove(B_ball)

        # generate balls to MAX_BALL_NUM
        for _, val in enumerate(_new_ball_types):
            self.balls.append(self.randball(np.int(val)))

        self.state = np.vstack([ball.state() for (_, ball) in enumerate(self.balls)])

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # create viewer
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W * self.scale, VIEWPORT_H * self.scale)

        # add ball to viewer
        for item in self.state:
            _x, _y, _s, _t = item[0] * self.scale, item[1] * self.scale, item[2], item[3]

            transform = rendering.Transform()
            transform.set_translation(_x, _y)

            # add a circle
            # center: (x, y)
            # radius: sqrt(score/pi)
            # colors: self in red, other in blue
            self.viewer.draw_circle(math.sqrt(_s / math.pi) * self.scale, 30, color=(_t, 0, 1)).add_attr(transform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        return

    @staticmethod
    def randball(_t: np.int):
        _b = Ball(np.random.rand(1)[0]*VIEWPORT_W, np.random.rand(1)[0]*VIEWPORT_H, np.random.rand(1)[0] * MAX_BALL_SCORE, np.int(np.random.rand(1)[0] * 360), _t)
        return _b


if __name__ == '__main__':
    # env = BallEnv()
    #
    # while True:
    #     env.step(150)
    #     env.render()
    s = spaces.Box(low=np.array([[-1.0, -2.0], [-1.0, -2.0]]), high=np.array([[2.0, 4.0], [2.0, 4.0]]), dtype=np.float32)
    print(s.sample())

