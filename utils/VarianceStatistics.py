
import numpy as np


class WelfordsVariance():
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    def __init__(self, shape):
        self.existingAggregate = (0, np.zeros(shape), np.zeros(shape))


    def update(self, newValue):
        # For a new value newValue, compute the new count, new mean, the new M2.
        # mean accumulates the mean of the entire dataset
        # M2 aggregates the squared distance from the mean
        # count aggregates the number of samples seen so far
        (count, mean, M2) = self.existingAggregate
        
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        
        self.existingAggregate = (count, mean, M2)


    def finalize(self):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = self.existingAggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)
    

class WelfordsVariance3():
    def __init__(self, shape):
        self.wv_eta = WelfordsVariance(shape)
        self.wv_hu = WelfordsVariance(shape)
        self.wv_hv = WelfordsVariance(shape)

    def update(self, eta, hu, hv):
        self.wv_eta.update(eta)
        self.wv_hu.update(hu)
        self.wv_hv.update(hv)

    def finalize(self, m=1):
        # m - values in [0, 1, 2]
        # m = 0 : mean
        # m = 1 : variance
        # m = 2 : sample variance
        return (self.wv_eta.finalize()[m], self.wv_hu.finalize()[m], self.wv_hv.finalize()[m])