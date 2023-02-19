import numpy as np

class Hopfield_Net:    
  
  def __init__(self, input):
    
    self.memory = np.array(input)
    self.n = self.memory.shape[1]
    self.state = np.random.randint(-2, 2, (self.n, 1))
    self.weights = np.zeros((self.n, self.n))
    self.energies = []

  def training_hebbian(self):
    self.weights = (1 / self.memory.shape[0]) * self.memory.T @ self.memory
    np.fill_diagonal(self.weights, 0)
  
  def update_state(self, n_to_update):
    for _ in range(n_to_update):
      index = np.random.randint(0, self.n)
      activation = np.dot(self.weights[index, :], self.state)
      if activation > 0:
        self.state[index] = 1
      else:
        self.state[index] = -1
      
  def compute_energy(self):
    self.energies.append(-0.5*np.dot(np.dot(self.state.T, self.weights), self.state))