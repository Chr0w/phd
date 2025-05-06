from pynput import keyboard
import cv2
import numpy as np
import os

# Constants
AGENT_WINDOW_NAME = "Agent Movement"
CONTOUR_WINDOW_NAME = "Contour Map"
MENU_WINDOW_NAME = "Map Selection"
INFO_WINDOW_NAME = "Info Window"
IMAGE_SIZE = 500
AGENT_RADIUS = 10
LASER_POINT_RADIUS = 3
AGENT_COLOR = (255, 0, 0)  # Blue in BGR
LASER_COLOR = (70, 70, 255)
MOVE_SPEED = 5
WHITE_COLOR = (255, 255, 255)  # White in BGR
MAPPED_COLOR = (120, 20, 30)
UNMAPPED_COLOR = (220, 170, 80)
MISSING_COLOR = (190, 110, 255)

HIGHLIGHT_COLOR = (0, 255, 0)  # Green for highlighting the selected map in the menu

# Initialize agent position
agent_x = IMAGE_SIZE // 2
agent_y = IMAGE_SIZE // 2

# Initialize movement flags
move_up = move_down = move_left = move_right = False


class sample_collection:
    def __init__(self):
        self.samples = []

    def add(self, sample):
        self.samples.append(sample)

    def to_dict(self):
        return [samp.__dict__ for samp in self.samples]

class sample:
    def __init__(self, origin=None):
        self.origin = origin
        self.score = 0
        self.score_normalized = 0
        self.points = []
        self.kd_tree = None

    def add_point(self, point):
        self.points.append(point)

    def to_dict(self):
        return {
            'origin': self.origin.to_dict() if self.origin else None,
            'score': self.score,
            'points': [point.to_dict() for point in self.points]
        }

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dist_to_nearest_neighbor = None
        self.inlier = False

    def to_tuple(self):
        return (self.x, self.y)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'dist_to_nearest_neighbor': self.dist_to_nearest_neighbor
        }
    
    def __str__(self):
        return f"{self.x},{self.y}"


def sample_visible_objects(image, position, detection_colors, missing_color, angle_step=5):
    """
    Sample visible objects in the image from the given position.
    Uses a more efficient approach with reduced angular steps and early stopping.
    """
    height, width, _ = image.shape
    samp = sample(origin=point(x=position.x, y=position.y))
    missing_points = []

    # Iterate through angles with a step size
    for angle in range(0, 360, angle_step):
        #print("angle:", angle)
        radian = np.deg2rad(angle)
        dx = np.cos(radian)
        dy = np.sin(radian)
        #print("dx,dy:", dx, dy)

        # Iterate outward from the center point
        x_float, y_float = float(position.x), float(position.y)
        while 0 <= x_float < width and 0 <= y_float < height:
            x_float = x_float + dx
            y_float = y_float + dy

            x = int(x_float)
            y = int(y_float)

            #print("x,y:", x, y)
            if x < 0 or x >= width or y < 0 or y >= height:
                #print("break")
                break

            if np.array_equal(image[y, x], missing_color):
                missing_points.append(point(x=x, y=y))


            # Check if the pixel is non-white
            if any(np.array_equal(image[y, x], color) for color in detection_colors.values()):
                x += round(np.random.normal(0,1))
                y += round(np.random.normal(0,0.5))
                samp.add_point(point(x=x, y=y))
                break  # Stop further sampling in this direction

    return samp, missing_points

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

    # Close the menu window after a map is selected
    cv2.destroyWindow(MENU_WINDOW_NAME)

    return map_paths[selected_index]

def is_white(pixel):
    """Check if a pixel is white."""
    return np.array_equal(pixel, WHITE_COLOR)

def create_contour_map(image):
    """Create a contour map from the given image with contours having the same colors as the original items."""
    # Define the known object colors

    known_colors = {
        "mapped": MAPPED_COLOR,
        "unmapped": UNMAPPED_COLOR,
        "missing": MISSING_COLOR
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

def get_agent_position(game_map):

    global agent_x, agent_y
    # Update agent position based on movement flags
    if move_up and is_white(game_map[agent_y - MOVE_SPEED, agent_x]):
        agent_y = max(agent_y - MOVE_SPEED, AGENT_RADIUS)
    if move_down and is_white(game_map[agent_y + MOVE_SPEED, agent_x]):
        agent_y = min(agent_y + MOVE_SPEED, IMAGE_SIZE - AGENT_RADIUS)
    if move_left and is_white(game_map[agent_y, agent_x - MOVE_SPEED]):
        agent_x = max(agent_x - MOVE_SPEED, AGENT_RADIUS)
    if move_right and is_white(game_map[agent_y, agent_x + MOVE_SPEED]):
        agent_x = min(agent_x + MOVE_SPEED, IMAGE_SIZE - AGENT_RADIUS)

    return agent_x, agent_y

# def update_info_window(agent_x, agent_y, additional_info=None):
#     """Update the information window with agent position and other values."""
#     info_window = np.full((200, 300, 3), WHITE_COLOR, dtype=np.uint8)  # Create a blank white window

#     # Display the agent's position
#     cv2.putText(info_window, "Agent Position (x,y):    ", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
#     cv2.putText(info_window, f"[{agent_x}, {agent_y}]", (150, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)
#     # cv2.putText(info_window, f"Y: {agent_y}", (150, 60), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1)

#     # Display additional information if provided
#     if additional_info:
#         y_offset = 90
#         for key, value in additional_info.items():
#             cv2.putText(info_window, f"{key}:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
#             cv2.putText(info_window, str(value), (150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
#             y_offset += 30

#     # Show the information window
#     cv2.imshow(INFO_WINDOW_NAME, info_window)


def update_info_window(agent_x, agent_y, counted_colors, additional_info=None):
    """Update the information window with agent position, color counts, and other values."""
    info_window = np.full((300, 300, 3), WHITE_COLOR, dtype=np.uint8)  # Create a blank white window

    # Display the agent's position
    cv2.putText(info_window, "Agent Position (x, y):", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_window, f"[{agent_x}, {agent_y}]", (180, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Display the counted colors
    y_offset = 60
    for color, count in counted_colors.items():
        color_text = f"Color {color}:"
        cv2.putText(info_window, color_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(info_window, str(count), (180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_offset += 30

    # Display additional information if provided
    if additional_info:
        for key, value in additional_info.items():
            cv2.putText(info_window, f"{key}:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(info_window, str(value), (180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 30

    # Show the information window
    cv2.imshow(INFO_WINDOW_NAME, info_window)

def window_closed():
    return (cv2.getWindowProperty(AGENT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(CONTOUR_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(INFO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1)

def count_pixels_by_color(image, color_dict):
    """
    Count the number of pixels in the image for each color in the dictionary.

    Args:
        image (numpy.ndarray): The input image (H x W x 3).
        color_dict (dict): A dictionary where keys are colors (tuples) and values are counts (int).

    Returns:
        dict: The updated dictionary with counts of pixels for each color.
    """
    # Initialize counts to 0 for each color
    for color in color_dict:
        color_dict[color] = 0

    # Iterate through each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = tuple(image[y, x])  # Convert pixel to a tuple (R, G, B)
            if pixel in color_dict:
                color_dict[pixel] += 1

    return color_dict

def main():
    # Load maps
    map_folder = "edi_v2/test_maps"
    map_paths = load_maps(map_folder)

    # Display menu and get selected map
    selected_map_path = get_selected_map(map_paths)
    game_map = cv2.imread(selected_map_path)
    game_map = cv2.resize(game_map, (IMAGE_SIZE, IMAGE_SIZE))

    # Create contour map
    contour_map_org = create_contour_map(game_map)

    # Initialize agent position
    global agent_x, agent_y
    agent_x = IMAGE_SIZE // 2
    agent_y = IMAGE_SIZE // 2

    # Create windows
    cv2.namedWindow(AGENT_WINDOW_NAME)
    cv2.namedWindow(CONTOUR_WINDOW_NAME)
    cv2.namedWindow(INFO_WINDOW_NAME)

    # Position the windows side by side
    cv2.moveWindow(AGENT_WINDOW_NAME, 0, -500)  # Position the Contour Map window to the right
    cv2.moveWindow(CONTOUR_WINDOW_NAME, 600, 0)  # Position the Agent Movement window at (0, 0)
    cv2.moveWindow(INFO_WINDOW_NAME, 1150, 0)  # Position the Info Window to the right of Agent Movement

    # Start listening to keyboard events
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    img_height, img_width, _ = contour_map_org.shape

    # main loop
    while True:
        if window_closed():
            break
        
        # Clear the image
        image = game_map.copy()
        contour_map = contour_map_org.copy()

        # Update agent position
        agent_x, agent_y = get_agent_position(game_map)

        detection_colors = {
        "mapped": MAPPED_COLOR,
        "unmapped": UNMAPPED_COLOR,
    }  
        scan, missing_points = sample_visible_objects(contour_map, point(agent_x, agent_y), detection_colors=detection_colors, missing_color=MISSING_COLOR, angle_step=3)

        for p in missing_points:
            cv2.rectangle(contour_map, (p.x, p.y), (min(p.x+1, img_width-1), min(p.y+1, img_height-1)), HIGHLIGHT_COLOR, 2)
        contour_map_org = contour_map.copy()

        # Update the contour map with the laser points
        for p in scan.points:
            cv2.circle(contour_map, (p.x, p.y), LASER_POINT_RADIUS, LASER_COLOR, 2)

        # Draw the agent
        cv2.circle(image, (agent_x, agent_y), AGENT_RADIUS, AGENT_COLOR, 1)

        counted_colors = {
            MAPPED_COLOR: 0,
            UNMAPPED_COLOR: 0,
            MISSING_COLOR: 0
        }

        counted_colors = count_pixels_by_color(contour_map, counted_colors)

        # Display the images
        image = cv2.resize(image, (1000, 1000))
        contour_map = cv2.resize(contour_map, (1000, 1000))

        cv2.imshow(AGENT_WINDOW_NAME, image)
        cv2.imshow(CONTOUR_WINDOW_NAME, contour_map)
        update_info_window(agent_x, agent_y, counted_colors)

        # Exit on 'q' key
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()
    listener.stop()

if __name__ == "__main__":
    main()