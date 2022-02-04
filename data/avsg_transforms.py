import numpy as np

#########################################################################################

class set_actors_num_and_shuffle(object):
    """
    take the closest max_num_agents agents to the ego (the agents are ordered this way in the data)
    also shuffle their indexing
    """
    def __init__(self, opt):
        self.max_num_agents = opt.max_num_agents

    def __call__(self, sample):
        agents_feat = sample['agents_feat']
        agents_num_orig = agents_feat['agents_num']
        agents_num = min(agents_num_orig, self.max_num_agents)
        closest_agents_inds = np.arange(agents_num)
        inds = np.random.shuffle(closest_agents_inds)
        sample['agents_feat']['agents_num'] = agents_num
        sample['agents_feat']['agents_data'] = sample['agents_feat']['agents_data'][inds]
        return sample



#########################################################################################




