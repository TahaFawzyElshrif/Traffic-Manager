import random

class Memory:  
    def __init__(self, capacity, seed=0):
        # Set random seed if provided
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)

            
        self.capacity = capacity
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        # Ensure we're using the same seed for consistent sampling
        if self.seed_value is not None:
            random.seed(self.seed_value)
            
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)