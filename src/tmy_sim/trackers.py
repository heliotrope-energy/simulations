import simple_rl as rl

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
    def get_angle(self, state):
        return randint(self.min, self.max)
    def get_azimuth(self):
        return self.azimuth

class AstroTracker:
    def __init__(self, azimuth, fallback_angle=30):
        self.name="astronomical"
        self.azimuth = azimuth
        self.fallback_angle = fallback_angle
    def get_angle(self, state):

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

        if new_angle < limits[0] or new_angle > limits[1]:
            return prev_angle

        return angle
