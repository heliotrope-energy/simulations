import simple_rl as rl
import numpy as np
from energy_calcs import *
from tqdm import tqdm
from random import randint

class FixedPolicyTracker:
    def __init__(self, angle, azimuth):
        self.name="Fixed at {}".format(angle)
        self.angle = angle
        self.azimuth = azimuth
        self.fallback_angle = 0
    def get_angle(self, state):
        return self.angle
    def get_azimuth(self):
        return self.azimuth


class RandomTracker:
    def __init__(self, min, max, azimuth):
        self.name="Random from {} to {}".format(min, max)
        self.max = max
        self.min = min
        self.azimuth = azimuth
        self.fallback_angle= 0
    def get_angle(self, state, reward):
        return randint(self.min, self.max)
    def get_azimuth(self):
        return self.azimuth

class AstroTracker:
    def __init__(self, azimuth, fallback_angle=30):
        self.name="astronomical"
        self.azimuth = azimuth
        self.fallback_angle = fallback_angle
    def get_angle(self, state, reward):

        # angle_pos = pvlib.tracking.singleaxis(zenith, azi, backtrack=False)
        surface_tilt = state.get_objects_of_class('tracker')[0]['tracker_theta']
        # if np.isnan(surface_tilt[0]):
        #     #TODO: fix this
        #     surface_tilt = self.fallback_angle
        return surface_tilt
    def get_azimuth(self):
        return self.azimuth


class LinUCBTracker:
    def __init__(self, azimuth, context_size, limits = (-70, 70), alpha=0.3, bins=25):
        self.name="lin-ucb"
        self.azimuth = azimuth
        self.alpha = alpha
        self.context_size = context_size
        self.fallback_angle = 30

        self.angles = np.linspace(limits[0], limits[1], num=bins)

        self.action_dict = {str(angle):angle for angle in self.angles}
        self.prev_reward = 0

        self.actions = list(self.action_dict.keys())
        #initialize learning agent
        self.agent = rl.agents.LinUCBAgent(self.actions, context_size = context_size, alpha=alpha)

    def get_azimuth(self):
        return self.azimuth

    def get_angle(self, context, prev_reward):
        '''
        Uses RL agent!
        '''
        action = self.agent.act(context, prev_reward)

        angle = self.action_dict[action]
        return angle

#TODO: experiment with action space for both stepwise and bandit actions
#TODO: test with delta reward
class SARSATracker:
    def __init__(self, azimuth, num_features, limits = (-70, 70), action_step = 5):
        self.name="SARSA"
        self.azimuth = azimuth
        self.fallback_angle = 30
        self.action_step = action_step
        self.limits = limits

        actions = ['inc', 'dec', 'same']
        self.agent = rl.agents.LinearSarsaAgent(actions, num_features)

    def get_azimuth(self):
        return self.azimuth

    def get_angle(self, state, prev_reward):
        '''
        Uses RL agent!
        '''
        action = self.agent.act(state, prev_reward)

        prev_angle = state.get_objects_of_class('tracker')[0]['prev_angle']
        new_angle = prev_angle
        if action == "inc":
            new_angle += self.action_step
        elif action == "dec":
            new_angle -= self.action_step

        if new_angle < self.limits[0] or new_angle > self.limits[1]:
            return prev_angle

        return new_angle

class OptimalTracker:
    '''
    Scans every possible angle for the best configuration.
    '''
    def __init__(self, azimuth, limits=(-90, 90), bins=100):
        self.name="Optimal"
        self.azimuth = azimuth
        self.fallback_angle = 30
        self.configurations = np.linspace(limits[0], limits[1], num=bins)

    def get_azimuth(self):
        return self.azimuth

    def get_angle(self, state, prev_reward):
        '''
        Slow and steady hopefully wins the race.
        '''

        current_index = pd.to_datetime(state.get_objects_of_class('env')[0]['datetime'])

        max_pwr = 0
        max_angle = 0
        for config in self.configurations:
            _,ac,  _, _ = calculate_energy(config,
                                        self.azimuth,
                                        state.get_objects_of_class('env')[0]['albedo'],
                                        state.get_objects_of_class('env')[0]['Wspd'],
                                        state.get_objects_of_class('env')[0]['DryBulb'],
                                        current_index, state.get_objects_of_class('sun')[0], state.get_objects_of_class('env')[0]['DHI'], state.get_objects_of_class('env')[0]['DNI'], state.get_objects_of_class('env')[0]['GHI'], "", save_data=False)

            if float(ac) > max_pwr:
                max_pwr = float(ac)
                max_angle = config

        return max_angle
