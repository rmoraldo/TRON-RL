from Environment import GridWorldEnv
import sys
import numpy as np
import pygame

def main():

    pygame.init()

    env = GridWorldEnv(size=32)
    env.render_mode = "human"
    obs, info = env.reset()

    running = True

    key_to_action = {
        pygame.K_RIGHT: 1, #(0,0) is top left of pygame grid -> (y,x) instead of (x,y)
        pygame.K_UP: 2,
        pygame.K_LEFT: 3,
        pygame.K_DOWN: 0,
    }
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Distance: {info['distance']}")
                    if terminated:
                        print("🎯 Target reached! Resetting environment.")
                        obs, info = env.reset()

        # Render
        env.render()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()