import pygame
import random
import numpy as np
import os

pygame.init()

WIDTH, HEIGHT = 400, 600
PIPE_WIDTH = 50
PIPE_GAP = 200
GRAVITY = 0.25
FLAP_FORCE = -9
BG_SPEED = 1

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")
font = pygame.font.Font(None, 36)
game_over_font = pygame.font.Font(None, 48)

bird_img = pygame.image.load('bird.jpeg')  
bird_img = pygame.transform.scale(bird_img, (30, 30))
pipe_img = pygame.image.load('pipe.jpeg')  
pipe_img = pygame.transform.scale(pipe_img, (PIPE_WIDTH, 150))
background_img = pygame.image.load('bg.jpg')
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))

class Bird:
    def __init__(self):
        self.x = 50
        self.y = HEIGHT // 2
        self.velocity = 0
        self.gravity = GRAVITY

    def flap(self):
        self.velocity = FLAP_FORCE

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def draw(self):
        win.blit(bird_img, (self.x, self.y))
    def off_screen(self):
        return self.y < -50 or self.y > 650
        

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.gap = PIPE_GAP
        self.height = random.randint(100, HEIGHT - self.gap - 100)

    def move(self):
        self.x -= 2

    def off_screen(self):
        return self.x < -PIPE_WIDTH

    def draw(self):
        win.blit(pipe_img, (self.x, 0))
        win.blit(pipe_img, (self.x, self.height + self.gap))

bird = Bird()
pipes = []
bgx=0
start_time = 0
game_state = "paused"
timer_font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()
run = True

state_space = (10, 10, 10) 
action_space = 2  
qtable_file = "qtable.npy"

if os.path.exists(qtable_file):
    q_table = np.load(qtable_file)
else:
    q_table = np.zeros(state_space + (action_space,))
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0 
epsilon_decay = 0.995
epsilon_min = 0.01

def get_state(bird, pipes):
    y_bin = min(int(bird.y / HEIGHT * state_space[0]), state_space[0] - 1)
    v_bin = min(int((bird.velocity + 10) / 20 * state_space[1]), state_space[1] - 1)
    if pipes:
        pipe = pipes[0]
        dist_bin = min(int((pipe.x - bird.x) / WIDTH * state_space[2]), state_space[2] - 1)
    else:
        dist_bin = state_space[2] - 1
    return (y_bin, v_bin, dist_bin)

def choose_action(state):
    global epsilon
    if np.random.rand() < epsilon:
        return np.random.randint(action_space)
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

def game_over_screen():
    game_over_text = game_over_font.render("Game Over", True, WHITE)
    restart_text = font.render("Press Space to Restart", True, WHITE)
    
    game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
    
    win.blit(game_over_text, game_over_rect)
    win.blit(restart_text, restart_rect)

def restart_game():
    global bird, pipes, start_time, game_state, epsilon
    bird = Bird()
    pipes = []
    start_time = 0
    game_state = "paused"
    epsilon=1.0
    np.save(qtable_file, q_table)

bird = Bird()
pipes = []
bgx = 0
start_time = 0
game_state = "paused"

clock = pygame.time.Clock()
run = True

while run:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if game_state == "paused":
                if event.key == pygame.K_SPACE:
                    game_state = "playing"
                    start_time = pygame.time.get_ticks() - 500 
            elif game_state == "playing":
                if event.key == pygame.K_SPACE:
                    bird.flap()
            elif game_state == "game_over":
                if event.key == pygame.K_SPACE:  
                    bird = Bird()
                    pipes = []
                    start_time = 0
                    game_state = "paused"
    bgx -= BG_SPEED
    if bgx <= -WIDTH:
        bgx = 0
  
    if game_state == "playing":
      
        state = get_state(bird, pipes)
        action = choose_action(state)
        
        if action == 1:
            bird.flap()
        
        bird.update()

        for pipe in pipes:
            pipe.move()

        if pipes and pipes[0].off_screen():
            pipes.pop(0)

        if len(pipes) < 3:
            pipes.append(Pipe())

        reward = 1  
        for pipe in pipes:
            if (bird.x + bird_img.get_width() > pipe.x and bird.x < pipe.x + PIPE_WIDTH and
                (bird.y < pipe.height or bird.y + bird_img.get_height() > pipe.height + pipe.gap)):
                reward = -1000  
                game_state = "game_over"
                break

        if bird.off_screen():
            reward = -1000 
            game_state = "game_over"
        
        next_state = get_state(bird, pipes)
        
        update_q_table(state, action, reward, next_state)
      
        win.fill(WHITE)
        win.blit(background_img, (bgx, 0))
        win.blit(background_img, (bgx + WIDTH, 0))
        bird.draw()
        for pipe in pipes:
         pipe.draw()
        elapsed_time = pygame.time.get_ticks() - start_time
        timer_text = timer_font.render("Time: " + str(elapsed_time // 1000), True, WHITE)
        win.blit(timer_text, (10, 10))

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    elif game_state == "game_over":
        win.fill(BLACK)  
        game_over_screen()
    
    elif game_state == "paused":
        paused_text = font.render("Press to Start", True, WHITE)
        paused_rect = paused_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        win.blit(paused_text, paused_rect)

    pygame.display.update()
np.save(qtable_file, q_table)
pygame.quit()