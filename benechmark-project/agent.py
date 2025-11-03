import numpy
import random
import math
from brain import Brain
from fullBrain import FullDqnBrain
from memory import Memory

gamma = 0.99
maxEpsilon = 1
minEpsilon = 0.01
lambda_ = 0.001

class Agent:
    steps = 0
    epsilon = maxEpsilon
    
    def __init__(self, id, stateCount, actionCount, seed=0):
        # Set random seed if provided
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

            
        self.stateCount = stateCount
        self.actionCount = actionCount
        self.id = id
        self.brain = Brain(stateCount, actionCount)
        self.memory = Memory(100000,self.seed_value)
    
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
           return numpy.argmax(self.brain.predicFlatten(s))
    
    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * math.exp(-lambda_ * self.steps)
    
    def replay(self):
        batch = self.memory.sample(64)
        batchLen = len(batch)
        
        no_state = numpy.zeros(self.stateCount)
        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        
        x = numpy.zeros((batchLen, self.stateCount))
        y = numpy.zeros((batchLen, self.actionCount))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            t = p[i]
            
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + gamma * numpy.amax(p_[i])
            
            x[i] = s
            y[i] = t
        
        self.brain.train(x, y)

class FullDqnAgent(Agent):
    def __init__(self, id, stateCount, actionCount, seed=None):
        # Set random seed if provided
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

            
        self.stateCount = stateCount
        self.actionCount = actionCount
        self.id = id
        self.brain = FullDqnBrain(stateCount, actionCount,self.seed_value)
        self.memory = Memory(100000,self.seed_value)
    
    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = minEpsilon + (maxEpsilon - minEpsilon) * math.exp(-lambda_ * self.steps)