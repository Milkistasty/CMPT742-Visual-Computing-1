import pygame
import math
import random

# Constants
WIDTH, HEIGHT = 800, 600
POSITIONS = ['UpLeft', 'UpMiddle', 'UpRight', 
             'MiddleLeft', 'MiddleMiddle', 'MiddleRight', 
             'BottomLeft', 'BottomMiddle', 'BottomRight']
STATES = ['Chill', 'Alert', 'Escape']
DISTANCE_THRESHOLDS = {'chill': 200, 'alert': 100}  # Distances in pixels

# Initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define cat grid positions
grid_positions = {
    'UpLeft': (WIDTH // 6, HEIGHT // 6),
    'UpMiddle': (WIDTH // 2, HEIGHT // 6),
    'UpRight': (5 * WIDTH // 6, HEIGHT // 6),
    'MiddleLeft': (WIDTH // 6, HEIGHT // 2),
    'MiddleMiddle': (WIDTH // 2, HEIGHT // 2),
    'MiddleRight': (5 * WIDTH // 6, HEIGHT // 2),
    'BottomLeft': (WIDTH // 6, 5 * HEIGHT // 6),
    'BottomMiddle': (WIDTH // 2, 5 * HEIGHT // 6),
    'BottomRight': (5 * WIDTH // 6, 5 * HEIGHT // 6)
}

# Cat class
class Cat:
    def __init__(self):
        self.position = 'MiddleMiddle'
        self.state = 'Chill'
        self.image = pygame.Surface((50, 50))  # Placeholder for the cat image
        self.rect = self.image.get_rect(center=grid_positions[self.position])

    def update(self, mouse_pos):
        distance = self.get_distance(mouse_pos)
        if distance > DISTANCE_THRESHOLDS['chill']:
            self.state = 'Chill'
        elif distance > DISTANCE_THRESHOLDS['alert']:
            self.state = 'Alert'
        else:
            self.state = 'Escape'
            self.change_position_based_on_mouse_direction(mouse_pos)

        # Update cat's image and rectangle for drawing
        self.image = self.load_cat_image()  # Load the appropriate image
        self.rect = self.image.get_rect(center=grid_positions[self.position])

    def get_distance(self, mouse_pos):
        cat_x, cat_y = grid_positions[self.position]
        mouse_x, mouse_y = mouse_pos
        return math.sqrt((cat_x - mouse_x) ** 2 + (cat_y - mouse_y) ** 2)

    def change_position_based_on_mouse_direction(self, mouse_pos):
        # Implement logic to change the cat's position based on the mouse's direction
        # For simplicity, let's just change the position randomly for now
        self.position = random.choice(POSITIONS)

    def load_cat_image(self):
        # Implement the method to load the appropriate image based on the cat's state and position
        # Placeholder implementation
        return pygame.Surface((50, 50))  # Placeholder for the cat image

    def draw(self, screen):
        screen.blit(self.image, self.rect)

# Main game loop
def game_loop():
    cat = Cat()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear screen with white color
        mouse_pos = pygame.mouse.get_pos()
        cat.update(mouse_pos)
        cat.draw(screen)

        pygame.display.flip()

game_loop()
pygame.quit()
