"""" This module implements the WGAN algorithm."""

import torchvision
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class DataManager:
    """ For managing batches of a data set."""
    def __init__(self, tensors):

        self.data = tensors
        self.curr = 0

    def get_batch(self, size):
        
        if self.curr + size > len(self.data):
            self.curr = 0

        return self.data[self.curr:self.curr+size]


class WganDiscriminator(nn.Module):

    def __init__(self, input_size, clipping, hidden_size=100):
        super(WganDiscriminator, self).__init__()
        
        """
        self.clayer1 = nn.Sequential( #28x28x1 input
            nn.Conv2d(1, 1, kernel_size=4, stride=1), #25x25 output size
            nn.ReLU()
        ) 
        """
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(hidden_size, 1))

        rv = torch.distributions.Normal(0, 0.1 * clipping)

        for m in self.modules():

            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                m.weight = nn.Parameter(rv.sample(m.weight.shape))
                m.bias = nn.Parameter(torch.zeros(m.bias.shape))


    def forward(self, x):
        """x = x.unsqueeze(1)
        out = self.clayer1(x)"""

        out = x.reshape(x.shape[0], -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        return out

class WganGenerator(nn.Module):

    def __init__(self, input_size, out_size, hidden_size=100):

        super(WganGenerator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.layer3 = nn.Linear(hidden_size, out_size)


        rv = torch.distributions.Normal(0, 0.1)

        for m in self.modules():

            if type(m) == nn.Linear:
                m.weight = nn.Parameter(rv.sample(m.weight.shape))
                m.bias = nn.Parameter(torch.zeros(m.bias.shape))

    def forward(self, x):
        out = x.reshape(x.shape[0], -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.shape[0], 28, 28)
        return out


class VanillaGenerator(nn.Module):

    def __init__(self, input_size, out_size, hidden_size=100):

        super(VanillaGenerator, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.layer3 = nn.Linear(hidden_size, out_size)


        rv = torch.distributions.Normal(0, 0.1)

        for m in self.modules():

            if type(m) == nn.Linear:
                m.weight = nn.Parameter(rv.sample(m.weight.shape))
                m.bias = nn.Parameter(torch.zeros(m.bias.shape))

    def forward(self, x):
        out = x.reshape(x.shape[0], -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.shape[0], 28, 28)
        return out


class VanillaDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size=100):
        super(VanillaDiscriminator, self).__init__()
        
        """
        self.clayer1 = nn.Sequential( #28x28x1 input
            nn.Conv2d(1, 1, kernel_size=4, stride=1), #25x25 output size
            nn.ReLU()
        ) 
        """
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        
        self.layer3 = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

        rv = torch.distributions.Normal(0, 0.1)

        for m in self.modules():

            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                m.weight = nn.Parameter(rv.sample(m.weight.shape))
                m.bias = nn.Parameter(torch.zeros(m.bias.shape))


    def forward(self, x):
        """x = x.unsqueeze(1)
        out = self.clayer1(x)"""

        out = x.reshape(x.shape[0], -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        return out


def clip_weights(m, c):

    if type(m) == nn.Linear:

        with torch.no_grad():
            m.weight.clamp_(-c, c)
            m.bias.clamp_(-c, c)

def wgan(generator, discriminator, data, noise, c, n_discr=100, n_iter=1000, batch_size=100):
    """ Implements the WGAN algorithm for training generative adversarial 
        networks.

        Args:
            generator: a neural network to use as the generator g_\theta, type nn.Module
            discriminator: a neural network to use as the discriminator f_w, type nn.Module
            data: an object which allows access to real data samples using the 
            interface data.get_batch(m), which returns m real data samples as a tensor
            noise: a probability distribution, eg from torch.distributions
            c: the clipping parameter, positive
            n_discr: the number of iterations to train the discriminator
            n_iter: the number of iterations to train the generator.
            batch_size: the number of samples to use each gradient update
        
        Returns:
            An optimized generator and discriminator
    """

    clipper = lambda m : clip_weights(m, c)

    #ensure weights are correctly clipped
    discriminator.apply(clipper)

    d_optimizer = torch.optim.RMSprop(discriminator.parameters())
    g_optimizer = torch.optim.RMSprop(generator.parameters())

    for _ in range(n_iter):

        for _ in range(n_discr):
            
            # see note below
            discriminator.requires_grad_(True)

            noise_sample = noise.sample((batch_size,))
            true_sample = data.get_batch(batch_size)

            with torch.no_grad():
                gen_sample = generator(noise_sample)

            d_optimizer.zero_grad()

            

            wloss = -(discriminator(true_sample).mean() - discriminator(gen_sample).mean())

            wloss.backward()

            d_optimizer.step()
            discriminator.apply(clipper)

        # I _think_ this line will avoid uneccesary computation of gradients
        # wrt the parameters of the discriminator (which is not being trained
        # in this section), but will still allow gradients to be computed through
        # the discriminator
        discriminator.requires_grad_(False)

        noise_sample = noise.sample((batch_size,))

        g_optimizer.zero_grad()

        out = - discriminator(generator(noise_sample)).mean()

        out.backward()
        g_optimizer.step()
        
    return generator, discriminator

def vanilla_gan(generator, discriminator, data, noise, c=None, n_discr=1, n_iter=10000, batch_size=100):

    d_optimizer = torch.optim.RMSprop(discriminator.parameters())
    g_optimizer = torch.optim.RMSprop(generator.parameters())

    z = noise.sample((1,))

    for _ in range(n_iter):


        if torch.any(torch.isnan(generator(z))):
            print(i)

            print(out)
            print(loss)        

        for _ in range(n_discr):
            
            # see note below
            discriminator.requires_grad_(True)

            noise_sample = noise.sample((batch_size,))
            true_sample = data.get_batch(batch_size)

            with torch.no_grad():
                gen_sample = generator(noise_sample)

            d_optimizer.zero_grad()

            

            loss = -(torch.log(discriminator(true_sample)) + (1 - torch.log(discriminator(gen_sample)))).mean()

            loss.backward()

            d_optimizer.step()

        # I _think_ this line will avoid uneccesary computation of gradients
        # wrt the parameters of the discriminator (which is not being trained
        # in this section), but will still allow gradients to be computed through
        # the discriminator
        discriminator.requires_grad_(False)

        noise_sample = noise.sample((batch_size,))

        g_optimizer.zero_grad()

        out = torch.log(1 - discriminator(generator(noise_sample))).mean()

        out.backward()
        g_optimizer.step()
        
    return generator, discriminator

if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)

    disc_in = len(trainset.data[0].flatten())

    gen_in = 64
    gen_out = disc_in

    print(disc_in)
    print()
    print(gen_in)
    print(gen_out)
    

    clipping = 1

    noise = torch.distributions.MultivariateNormal(torch.zeros(gen_in), torch.eye(gen_in))

    data_manager = DataManager(trainset.data[trainset.targets == 3] / 255)
 
    """
    Discriminator = WganDiscriminator
    Generator = WganGenerator
    gan = wgan
    """
    Discriminator = VanillaDiscriminator
    Generator = VanillaGenerator
    gan = vanilla_gan
    

    d = Discriminator(disc_in, clipping)
    g = Generator(gen_in, gen_out)


    g, d = gan(g, d, data_manager, noise, c=clipping)


    num_to_gen = 9
    fig = plt.figure()
    for i in range(9):

        
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(g(noise.sample((1,))).detach().numpy().reshape(28, 28), cmap=cm.Greys)

    plt.show()



