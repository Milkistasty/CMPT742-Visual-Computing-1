import pygame
import math
import cv2
from pygame.locals import QUIT
import os

# Constants
WIDTH, HEIGHT = 800, 600
POSITIONS = ['UpLeft', 'UpMiddle', 'UpRight',
             'MiddleLeft', 'MiddleMiddle', 'MiddleRight',
             'BottomLeft', 'BottomMiddle', 'BottomRight']
DISTANCE_THRESHOLDS = {'chill': 200, 'alert': 100}  # Distances in pixels

# Initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Cat class
class Cat:
    # Define grid_positions as a class-level variable
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

    def __init__(self):
        self.position = 'MiddleMiddle'
        self.state = 'Chill'
        self.image = pygame.Surface((50, 50))  # Placeholder for the cat image
        self.rect = self.image.get_rect(center=Cat.grid_positions[self.position])
        self.video_path = 'C:/Users/Alienware/Desktop/VC CMPT742/FinalProject/final_result/generated_frames_from_interp/output_video_after_interp.mp4'
        self.play_video = False

    def _get_distance(self, mouse_pos):
        cat_x, cat_y = Cat.grid_positions[self.position]
        mouse_x, mouse_y = mouse_pos
        return math.sqrt((cat_x - mouse_x) ** 2 + (cat_y - mouse_y) ** 2)

    def _change_position_based_on_mouse_direction(self, mouse_pos):
        # ... (same as before)
        return None

    def _load_cat_image(self):
        if self.play_video:
            return None  # Return None to indicate playing the video
        else:
            # Implement the method to load the appropriate image based on the cat's state and position
            # Placeholder implementation
            return pygame.Surface((50, 50))  # Placeholder for the cat image

    def draw(self, screen):
        if not self.play_video:
            screen.blit(self.image, self.rect)

    def update(self, mouse_pos):
        distance = self._get_distance(mouse_pos)
        if distance > DISTANCE_THRESHOLDS['chill']:
            self.state = 'Chill'
            self.play_video = False
        elif distance > DISTANCE_THRESHOLDS['alert']:
            self.state = 'Alert'
            self.play_video = False
        else:
            self.state = 'Escape'
            self.play_video = True
            self._change_position_based_on_mouse_direction(mouse_pos)

        # Load cat's image or play video
        if self.play_video:
            self.load_and_play_video()
        else:
            self.image = self._load_cat_image()  # Load the appropriate image
            self.rect = self.image.get_rect(center=Cat.grid_positions[self.position])

    def load_and_play_video(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate the frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(frame)

            screen.blit(frame, (0, 0))
            pygame.display.flip()

            # Add a delay to control the playback speed (adjust milliseconds as needed)
            pygame.time.delay(20)  # 100 milliseconds delay

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    cap.release()
                    os._exit(0)

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
