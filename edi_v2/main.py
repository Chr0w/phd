from pynput import keyboard
import cv2
import numpy as np
import os

# Constants
WINDOW_NAME = "Agent Movement"
CONTOUR_WINDOW_NAME = "Contour Map"
MENU_WINDOW_NAME = "Map Selection"
IMAGE_SIZE = 500
AGENT_RADIUS = 10
AGENT_COLOR = (255, 0, 0)  # Blue in BGR
MOVE_SPEED = 5
WHITE_COLOR = (255, 255, 255)  # White in BGR
HIGHLIGHT_COLOR = (0, 255, 0)  # Green for highlighting the selected map

# Initialize agent position
agent_x = IMAGE_SIZE // 2
agent_y = IMAGE_SIZE // 2

# Initialize movement flags
move_up = move_down = move_left = move_right = False

def on_press(key):
    global move_up, move_down, move_left, move_right
    try:
        if key == keyboard.Key.up:
            move_up = True
        elif key == keyboard.Key.down:
            move_down = True
        elif key == keyboard.Key.left:
            move_left = True
        elif key == keyboard.Key.right:
            move_right = True
    except AttributeError:
        pass

def on_release(key):
    global move_up, move_down, move_left, move_right
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        move_up = move_down = move_left = move_right = False

def load_maps(folder_path):
    """Load all map file paths from the folder."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def display_menu(map_paths, selected_index):
    """Display a menu with buttons for each map."""
    menu_image = np.full((500, 500, 3), WHITE_COLOR, dtype=np.uint8)
    button_height = 50
    for i, map_path in enumerate(map_paths):
        y_start = i * button_height
        y_end = y_start + button_height
        color = HIGHLIGHT_COLOR if i == selected_index else (200, 200, 200)
        cv2.rectangle(menu_image, (50, y_start + 10), (450, y_end - 10), color, -1)
        cv2.putText(menu_image, os.path.basename(map_path), (60, y_start + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow(MENU_WINDOW_NAME, menu_image)

def get_selected_map(map_paths):
    """Allow the user to navigate the menu and select a map."""
    selected_index = 0

    def on_press(key):
        nonlocal selected_index
        try:
            if key == keyboard.Key.up:
                selected_index = (selected_index - 1) % len(map_paths)
            elif key == keyboard.Key.down:
                selected_index = (selected_index + 1) % len(map_paths)
            elif key == keyboard.Key.enter:
                listener.stop()
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.running:
            display_menu(map_paths, selected_index)
            cv2.waitKey(100)  # Refresh the menu periodically

    return map_paths[selected_index]

def is_white(pixel):
    """Check if a pixel is white."""
    return np.array_equal(pixel, WHITE_COLOR)

def create_contour_map(image):
    """Create a contour map from the given image with contours having the same colors as the original items."""
    # Define the known object colors
    known_colors = {
        "mapped": (120, 20, 30),
        "unmapped": (220, 170, 80),
        "missing": (190, 110, 255),
    }
    color_tolerance = 20  # Allowable deviation for each color channel

    # Initialize the contour map with a white background
    contour_map = np.full_like(image, WHITE_COLOR, dtype=np.uint8)

    for label, color in known_colors.items():
        # Create a mask for the current color with tolerance
        lower_bound = np.array([max(0, c - color_tolerance) for c in color], dtype=np.uint8)
        upper_bound = np.array([min(255, c + color_tolerance) for c in color], dtype=np.uint8)
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Find contours for the current mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the objects with white and draw the contours with the original color
        for contour in contours:
            cv2.drawContours(contour_map, [contour], -1, WHITE_COLOR, -1)  # Fill with white
            cv2.drawContours(contour_map, [contour], -1, color, 1)  # Draw the contour with the original color

    return contour_map

def main():
    # Load maps
    map_folder = "edi_v2/test_maps"
    map_paths = load_maps(map_folder)

    # Display menu and get selected map
    selected_map_path = get_selected_map(map_paths)
    game_map = cv2.imread(selected_map_path)
    game_map = cv2.resize(game_map, (IMAGE_SIZE, IMAGE_SIZE))

    # Create contour map
    contour_map = create_contour_map(game_map)

    # Initialize agent position
    global agent_x, agent_y
    agent_x = IMAGE_SIZE // 2
    agent_y = IMAGE_SIZE // 2

    # Create windows
    cv2.namedWindow(WINDOW_NAME)
    cv2.namedWindow(CONTOUR_WINDOW_NAME)

    # Start listening to keyboard events
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while True:
        # Check if the windows are still open
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(CONTOUR_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Clear the image
        image = game_map.copy()

        # Update agent position based on movement flags
        if move_up and is_white(game_map[agent_y - MOVE_SPEED, agent_x]):
            agent_y = max(agent_y - MOVE_SPEED, AGENT_RADIUS)
        if move_down and is_white(game_map[agent_y + MOVE_SPEED, agent_x]):
            agent_y = min(agent_y + MOVE_SPEED, IMAGE_SIZE - AGENT_RADIUS)
        if move_left and is_white(game_map[agent_y, agent_x - MOVE_SPEED]):
            agent_x = max(agent_x - MOVE_SPEED, AGENT_RADIUS)
        if move_right and is_white(game_map[agent_y, agent_x + MOVE_SPEED]):
            agent_x = min(agent_x + MOVE_SPEED, IMAGE_SIZE - AGENT_RADIUS)

        # Draw the agent
        cv2.circle(image, (agent_x, agent_y), AGENT_RADIUS, AGENT_COLOR, -1)

        # Display the images
        cv2.imshow(WINDOW_NAME, image)
        cv2.imshow(CONTOUR_WINDOW_NAME, contour_map)

        # Exit on 'q' key
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()