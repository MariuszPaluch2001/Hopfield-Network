import requests, gzip, os, hashlib
import pygame
import numpy as np
import matplotlib.pyplot as plt

from hopfield_net import Hopfield_Net

URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"

def get_MNIST(url):
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


if __name__ == "__main__":
  X = get_MNIST(URL)[16:].reshape((-1, 784))
  X_binary = np.where(X > 20, 1, -1)
  memories_list = [X_binary[np.random.randint(len(X))]]

  net = Hopfield_Net(memories_list)
  net.training_hebbian()

  size = 20
  pygame.init()
  surface = pygame.display.set_mode((28*size, 28*size))
  pygame.display.set_caption("  ")

  running = True

  while running:

    cells = net.state.reshape(28,28).T

    surface.fill((211,211,211))
    
    for r, c in np.ndindex(cells.shape):
      if cells[r,c] == -1:
        col = (135, 206, 250)
      elif cells[r,c] == 1:
        col = (0,0,128)
      else:
        col = (255,140,0)
      
      pygame.draw.rect(surface, col, (r*size, c*size, size, size))

    net.update_state(16)
    net.compute_energy()
    pygame.display.update()
    pygame.time.wait(50)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

        plt.figure("weights", figsize=(10,7))
        plt.imshow(net.weights, cmap="RdPu")
        plt.xlabel("Weights of neurons")
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        plt.figure("Energy", figsize=(10,7))
        x = np.arange(len(net.energies))
        plt.scatter(x, np.array(net.energies))
        plt.xlabel("Generation")
        plt.ylabel("Energy")
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        pygame.quit()
plt.show()