import numpy as np

#########################################################################################

class set_actors_num_and_shuffle(object):
    """

    """

    def __init__(self, opt):
        self.max_num_agents = opt.max_num_agents

    def __call__(self, sample):
        agents_feat = sample['agents_feat']
        agents_num_orig = agents_feat['agents_num']
        inds = np.random.shuffle(np.arange(agents_num_orig))[:self.max_num_agents]  # take the closest agent to the ego
        sample['agents_feat']['agents_num'] = len(inds)
        sample['agents_feat']['agents_data'] = sample['agents_feat']['agents_data'][inds]
        return sample



#########################################################################################




