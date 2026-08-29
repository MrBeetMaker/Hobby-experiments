"""This module combines a number classifier with a drawing board and a panel that shows you the
class probabilities as well as the identified number.

Future works:
 * Center and crop number for uniform classification.
 * Expand to letters.
 * Compare to traditional AI models.

     - Made in January 2024."""
from PIL import Image
import torch
import torchvision.transforms as transforms
import pygame
import numpy as np
import sys
from number_identifier import load_model, Net

# Global variables
width, height = 280, 280

def center_of_gravity(image_array) -> tuple[int, int]:
    """Calculate the center of gravity for the non-zero pixel values"""

    rows, cols = np.nonzero(image_array)
    if len(rows) == 0:
        center_row = image_array.shape[0] // 2
    else:
        center_row = int(np.mean(rows))
    if len(cols) == 0:
        center_col = image_array.shape[1] // 2
    else:
        center_col = int(np.mean(cols))

    return center_row, center_col

def preprocess_image(drawing_board):
    """Returns the drawing board as a torch.tensor suitable as
    input image for the CNN."""

    bw_image = np.mean(drawing_board, axis=2)

    bw_image = np.rot90(bw_image, k= 1)     # Rotate 270 degrees
    bw_image = np.flip(bw_image, axis= 0)   # Flip vertical

    resized_image = np.array(Image.fromarray(bw_image).resize((28, 28)))

    min_val, max_val = np.min(resized_image), np.max(resized_image)
    normalized_image = 2 * ((resized_image - min_val) / max((max_val - min_val), 1)) - 1

    # Convert the resized image to a PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor(),])

    tensor_image = transform(normalized_image)

    return tensor_image.unsqueeze(0)

def classify_number(input_image, cnn, probabilities = False):
    """Classifies the input image.
    
    Parameters:
        - input_image: torch.tensor input image.
        - cnn: nn.Module CNN number classifier.
        - probabilities: Optional bool, set to true to return class probabilities as a second return value.
    """

    output = cnn(input_image)
    _, predicted = torch.max(output, 1)
    predicted = int(predicted)

    if probabilities:
        output = output[0]
        output = 1 - (output - max(output)) / min(output)
        return predicted, output
    return predicted

def handle_draw(drawing_board, x, y, thickness):
    """Updates the drawing board in-place."""
    x_1, y_1 = x + thickness % width, y + thickness % height
    drawing_board[x: x_1, y: y_1] = (255, 255, 255)

def main():

    # Set up drawing board
    thickness = 16
    drawing_board = np.zeros((width, height, 3))
    time_before_auto_preprocess = 400      # Miliseconds
    last_edit_time = 0
    image_is_ready = False

    # Initialize Pygame
    pygame.init()

    # Set up side bar
    side_width = width // 2
    sidebar = pygame.Surface((side_width, height))

    # Set up the display
    pygame.display.set_caption("Interactive Language Classifier")
    main_window = pygame.display.set_mode((width + side_width, height))
    screen = pygame.Surface((width, height))

    # Set up colors
    background_color = (0, 0, 0)
    sidebar_color = (60, 65, 55)

    # Set up buttons
    nr_of_boxes = 10

    h_pad = 3
    w_pad = 3

    box_h = height/nr_of_boxes - h_pad
    box_w = side_width - 2 * w_pad

    num_boxes = list()
    for i in range(nr_of_boxes - 1):
        num_boxes.append(pygame.Rect(w_pad, h_pad + i*height/nr_of_boxes, box_w, box_h))
    num_boxes.append(pygame.Rect(w_pad, h_pad + 9*height/nr_of_boxes, box_w, box_h - h_pad))

    # Prepare font
    font_path = pygame.font.match_font('Courier New')
    font_size = 24
    font = pygame.font.Font(font_path, font_size)
    text_color = (0, 20, 0)

    bold_font = pygame.font.Font(font_path, font_size)
    bold_font.set_bold(True)

    # Load number classifier.
    cnn = load_model()
    probabilities = [0.5 for _ in range(nr_of_boxes)]
    guess = -1

    # Main game loop
    fps = 1000
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:

                    if not image_is_ready:
                        input_image = preprocess_image(drawing_board)
                    guess, probabilities = classify_number(input_image, cnn, probabilities= True)

                elif event.key == pygame.K_SPACE:
                    drawing_board = np.zeros((width, height, 3))
                    guess = -1

                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Clear the screen
        screen.fill(background_color)
        sidebar.fill(sidebar_color)

        if any(pygame.mouse.get_pressed()):
            x, y = pygame.mouse.get_pos()

            if x < width and y < height:
                handle_draw(drawing_board, x, y, thickness)
                last_edit_time = pygame.time.get_ticks()
                image_is_ready = False

        if guess < 0:
            probabilities = [0.5 for _ in range(nr_of_boxes)]

        for box, val in zip(num_boxes, probabilities):
            color = [85 + int(val * 170), 100 + int(val * 155), 95 + int(val * 160)]
            pygame.draw.rect(sidebar, color, box)

        for i, box in enumerate(num_boxes):
            if i == guess:
                text_surface = bold_font.render(f"{i}", True, text_color)
            else:
                text_surface = font.render(f"{i}", True, text_color)
            sidebar.blit(text_surface, box.midtop)

        if not image_is_ready and pygame.time.get_ticks() - last_edit_time >= time_before_auto_preprocess:
            input_image = preprocess_image(drawing_board)
            image_is_ready = True

        pygame.surfarray.blit_array(screen, drawing_board)
        main_window.blit(screen, (0, 0))        # Blit the screen onto the main window
        main_window.blit(sidebar, (width, 0))   # Blit the sidebar onto the main window

        # Update the display
        pygame.display.flip()

        # Control the frame rate (optional)
        pygame.time.Clock().tick(fps)  # Adjust the frame rate as needed

    # Quit Pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":

    print("\nWelcome! Draw a number in on the board!\n\n\t - Press ENTER to let the AI guess the number.\n\t - Press SPACE to reset the board.\n\t - Press ESC to exit the program.\n\nTip: Draw the number in the center, and leave a bit of space around it.\n     That'll help the AI recognize the numbers better.\n")

    main()
