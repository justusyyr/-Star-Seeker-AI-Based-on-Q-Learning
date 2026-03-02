import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#1. DQN Architecture 
class DeepQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

#2. Advanced Star Seeker Environment (Optimized) 
class StarSeekerEnv:
    def __init__(self):
        self.w, self.h = 600, 600
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Star Seeker AI - Final Version")
        self.font = pygame.font.SysFont("arial", 20)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.player_pos = np.array([300.0, 300.0])
        self.target_pos = self._rand_pos()
        # Challenge 2: Reduced obstacle speed for easier initial learning
        self.obstacles = [self._rand_pos() for _ in range(3)]
        self.obs_dirs = [np.array([random.uniform(-1,1), random.uniform(-1,1)]) for _ in range(3)]
        
        self.energy = 100.0
        self.score = 0
        # Initialize distance for Challenge 1 (Guiding Reward)
        self.prev_dist = np.linalg.norm(self.player_pos - self.target_pos)
        return self.get_state()

    def _rand_pos(self):
        return np.array([random.randint(50, 550), random.randint(50, 550)], dtype=float)

    def get_state(self):
        dist_target = (self.target_pos - self.player_pos) / 600
        obs_rel = []
        for obs in self.obstacles:
            obs_rel.extend((obs - self.player_pos) / 600)
        state = [self.player_pos[0]/600, self.player_pos[1]/600] + list(dist_target) + obs_rel
        return np.array(state, dtype=np.float32)

    def step(self, action):
        move = np.array([[0,-5],[0,5],[-5,0],[5,0]])[action]
        self.player_pos += move
        self.energy -= 0.1 
        
        for i in range(len(self.obstacles)):
            # Slightly slower obstacles to help agent survive longer
            self.obstacles[i] += self.obs_dirs[i] * 1.5 
            if self.obstacles[i][0] < 0 or self.obstacles[i][0] > 600: self.obs_dirs[i][0] *= -1
            if self.obstacles[i][1] < 0 or self.obstacles[i][1] > 600: self.obs_dirs[i][1] *= -1

        reward = 0
        done = False

        #CHALLENGE 1: Guiding Reward (Potential Reward) 
        current_dist = np.linalg.norm(self.player_pos - self.target_pos)
        if current_dist < self.prev_dist:
            reward += 1.0  # Reward for moving closer
        else:
            reward -= 1.0  # Penalty for moving away
        self.prev_dist = current_dist

        #CHALLENGE 3: Larger Detection Radius 
        if current_dist < 30: # Increased from 20 to 30 to solve sparse reward
            reward = 20.0 
            self.score += 1
            self.energy = min(100, self.energy + 40)
            self.target_pos = self._rand_pos()
            self.prev_dist = np.linalg.norm(self.player_pos - self.target_pos)

        # Collision with obstacles
        for obs in self.obstacles:
            if np.linalg.norm(self.player_pos - obs) < 25:
                reward = -15.0
                done = True

        # Out of bounds
        if not (0 < self.player_pos[0] < 600 and 0 < self.player_pos[1] < 600):
            reward = -10.0
            done = True
            
        if self.energy <= 0: done = True

        return self.get_state(), reward, done

    def render(self, ep, epsilon):
        self.screen.fill((10, 10, 30))
        pygame.draw.circle(self.screen, (255, 215, 0), self.target_pos.astype(int), 12) # Gold Target
        for obs in self.obstacles:
            pygame.draw.circle(self.screen, (160, 32, 240), obs.astype(int), 15) # Purple Obstacle
        pygame.draw.rect(self.screen, (0, 191, 255), (self.player_pos[0]-12, self.player_pos[1]-12, 24, 24)) # Agent
        
        stats = self.font.render(f"Ep: {ep}  Score: {self.score}  Energy: {int(self.energy)}  Eps: {epsilon:.2f}", True, (255,255,255))
        self.screen.blit(stats, (10, 10))
        pygame.display.flip()

#3. Training Loop 
def train():
    pygame.init()
    env = StarSeekerEnv()
    agent = DeepQNet(10, 4)
    optimizer = optim.Adam(agent.parameters(), lr=0.0005)
    memory = deque(maxlen=20000)
    
    eps, eps_decay, eps_min = 1.0, 0.996, 0.05
    gamma = 0.98 

    for ep in range(600): # Increased episodes for better convergence
        state = env.reset()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return

            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = agent(torch.FloatTensor(state)).argmax().item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if ep % 5 == 0:
                env.render(ep, eps)
                env.clock.tick(60)

            if len(memory) > 64:
                batch = random.sample(memory, 64)
                s, a, r, ns, d = zip(*batch)
                s = torch.FloatTensor(np.array(s))
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                ns = torch.FloatTensor(np.array(ns))
                d = torch.FloatTensor(d)

                q_vals = agent(s).gather(1, a.unsqueeze(1))
                max_next_q = agent(ns).max(1)[0].detach()
                targets = r + (1 - d) * gamma * max_next_q

                loss = nn.MSELoss()(q_vals.squeeze(), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done: break
        
        eps = max(eps_min, eps * eps_decay)
        if ep % 10 == 0:
            print(f"Episode {ep} | Final Score: {env.score}")

    torch.save(agent.state_dict(), "star_seeker_final.pth")
    pygame.quit()

if __name__ == "__main__":
    train()