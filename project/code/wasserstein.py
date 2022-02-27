import torch 
from torch import nn
from scipy import stats
import numpy as np


class DefaultWDiscriminator(nn.Module):
    """ Default discriminator network for computing wasserstein distance"""
    def __init__(self, input_size, clipping):
        super(DefaultWDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, 10), nn.Sigmoid())
        self.layer2 = nn.Linear(10, 1)

        rv = torch.distributions.Normal(0, 0.1 * clipping)

        for m in self.modules():

            if type(m) == nn.Linear:
                m.weight = nn.Parameter(rv.sample(m.weight.shape))
                m.bias = nn.Parameter(torch.zeros(m.bias.shape))


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


def wasserstein_distance(dist1, dist2, sample_dim, batch_size=100, num_iter=500, c=0.5, discr=None):
    """ Computes the Wasserstein distance between two distributions.
        
        Args:
            dist1: the first distribution. Must implement the method dist1.sample(n) 
            where n is an integer which returns n vectors of dimension sample_dim
            dist2: the second distribution. As above.
            sample_dim: the dimension of the vectors produced by sample
    """

    def clipper(m):

        if type(m) == nn.Linear:
            #update the weights and biases in-place

            with torch.no_grad():
                m.weight.clamp_(-c, c)
                m.bias.clamp_(-c, c)


    if discr is None:
        discr = DefaultWDiscriminator(sample_dim, c)

    discr.apply(clipper)

    optimizer = torch.optim.RMSprop(discr.parameters(), lr=0.001)

    for _ in range(num_iter):
        
        sample1 = dist1.sample(batch_size)
        sample2 = dist2.sample(batch_size)

        optimizer.zero_grad()

        wass_loss = -(discr(sample1).mean() - discr(sample2).mean())
    
        wass_loss.backward()

        optimizer.step()
        discr.apply(clipper)

    return discr

class TestRV:
    """ A simple 1-dimensional distribution which can be sampled in the way 
        required by wasserstein_distance."""
    def __init__(self, theta):
        self.theta = theta

    def sample(self, n):
        return torch.from_numpy(np.vstack((self.theta * np.ones(n), stats.uniform.rvs(size=n))).T).to(torch.float)


    @staticmethod
    def dim():
        return 2

if __name__ == "__main__":


    num_vals = 21

    true_theta = 0

    thetas = np.linspace(-1, 1, num=num_vals)
    distance_w = np.empty(num_vals)

    distance_js = np.empty(num_vals)
    

    true_dist = TestRV(true_theta)

    print(true_dist.sample(1).numpy())

    for i, theta in enumerate(thetas):

        print(f"Processing {i}/{num_vals} ... ", end="")

        test_dist = TestRV(theta)

        d = wasserstein_distance(true_dist, test_dist, TestRV.dim())

        distance_w[i] = (d(true_dist.sample(1000)).mean() - d(test_dist.sample(1000)).mean()).detach().numpy()


        print("Done!")


    import matplotlib.pyplot as plt
    

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(thetas, distance_w, color='b')

    plt.show()

    print(thetas)
    print(distance_w)
    print(distance_js)


    