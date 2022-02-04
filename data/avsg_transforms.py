

#########################################################################################

class set_actors_num_and_shuffle(object):
    """

    """

    def __init__(self, opt):
        self.max_num_agents = opt.max_num_agents

    def __call__(self, sample):
        agents_feat = sample['agents_feat']
        agents_num_orig =
        #
        # inds = agents_dists_order[:self.max_num_agents]  # take the closest agent to the ego
        # np.random.shuffle(inds)  # shuffle so that the ego won't always be first
        sample['agents_feat'] = agents_feat
        return sample



#########################################################################################




