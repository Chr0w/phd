
import cv2
import numpy as np
from math import floor
from statistics import mean
import json


class map_point:
  def __init__(self, x, y, score=0):
    self.x = x
    self.y = y
    self.score = score

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def nearest_neighbor_distances(list1, list2):
    distances = []
    for point1 in list1:
        min_distance = float('inf')
        for point2 in list2:
            distance = euclidean_distance(point1, point2)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    return distances


def color_nearest_pixels(image, points, color, radius=1, white_only=False):

    # Color the nearest pixels around each point
    for point in points:
        if point:
            for y in range(point.y - radius, point.y + radius + 1):
                for x in range(point.x - radius, point.x + radius + 1):
                    # Check if the point is within the image bounds
                    if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
                        #if white_only and image[y, x] == 255:
                        image[y, x] = color
                        #if not white_only and image[y, x] != 255:
                        #    image[y, x] = color
    
    return image


def sample_visible_objects(image, points, detection_colors):
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # List to store the nearest non-white pixels
    nearest_pixels = []
    
    for p in points:
        # Iterate through 360 degrees
        for angle in range(360):
            radian = np.deg2rad(angle)
            
            # Iterate outward from the center point
            for radius in range(max(height, width)):
                x = int(p.x + radius * np.cos(radian))
                y = int(p.y + radius * np.sin(radian))
                
                # Check if the point is within the image bounds
                if x >= 0 and x < width and y >= 0 and y < height:
                    # Check if the pixel is non-white
                    #if gray_image[y, x] < 255:
                    if image[y, x] in detection_colors:
                            x += round(np.random.normal(0,1))
                            y += round(np.random.normal(0,0.5))
                            nearest_pixels.append(map_point(x=x, y=y))
                            break
            
    
    return nearest_pixels


    

def decode_color(color):

    white = [0, 0, 0]

    if color.all() == 0:
        print("white!")

def get_pixel_colors(image):

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Create a list to store pixel colors
    pixel_colors = []


    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            # Get the color of the pixel (BGR format)
            color = image[y, x]
            pixel_colors.append(color)

    return pixel_colors

def get_free_map_points(image, fraction):

    # Get the dimensions of the image
    height, width = image.shape

    # Create a list to store pixel colors
    map_points = []

    # Iterate through each pixel
    for y in range(0, height, floor(height/fraction)):
        for x in range(0, width, floor(width/fraction)):
            # Get the color of the pixel (BGR format)
            if image[y, x] == 255:
                map_points.append(map_point(x,y,0))

    return map_points

# Example usage


def display_result(image, display_size):
    # Display the image

    resized_image = cv2.resize(image, (display_size, display_size))
    cv2.imshow('Image', resized_image)

    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def min_max_normalize(data):
    # Calculate the minimum and maximum values in the list
    min_val = min(data)
    max_val = max(data)
    
    # Perform min-max normalization
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data


def remap(image):
    map_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(map_x.shape[0]):
        map_x[i,:] = [x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]

    return cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)


def load_image():
    image = cv2.imread('/home/mircrda/phd_main/edi/test_maps/map_1.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Could not open or find the image: ')
        exit(0)
    return image

def load_config():
    # Open and read the JSON file
    with open('config.json', 'r') as file:
        data = json.load(file)
    return data

def get_display_image(image, config):

    result = convert_to_bgr(image)
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            g_val = image[y, x]
            color_val = get_color_info(config=config, greyscale_value=g_val)
            if color_val == None:
                result[y, x] = [0, 160, 255] # Bright orange
            else:
                result[y, x] = color_val['BGR_value']

    return result


# def get_greyscale_value(config):
#     for c in get_color_info(config=config, greyscale_value=)

def convert_to_bgr(image):
    return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

def get_color_info(config, greyscale_value):
    return config["color_map"].get(str(greyscale_value), None)

def get_sample_points(sample_positions, image, config):

    #laser_points = []

    #for p in sample_positions:
        # print(f"robot pos: (", p.x, ",", p.y,")")
    laser_points = sample_visible_objects(image, sample_positions, detection_colors=[0, 2]) 

    return laser_points


def main():

    conf = load_config()
    map = load_image()

    sample_positions = get_free_map_points(image=map, fraction=5)

    # Refactor to keep the two data sets coupled with the sampling position.
    # Also, save the data so it can be loaded
    laser_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 2])
    map_sample = sample_visible_objects(map, sample_positions, detection_colors=[0, 3])


    map = color_nearest_pixels(map, sample_positions, 6)
    map = color_nearest_pixels(map, laser_sample, 4)
    map = color_nearest_pixels(map, map_sample, 5)

    display_image = get_display_image(map, conf)

    # Serializing json


    
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        json.dump(map_sample, outfile)


    display_result(display_image, conf["parameters"]["display_window_size"])

    # -----------------------------------------------------------------------
    exit()

    # Open an image file
    image = cv2.imread('/home/mircrda/phd_main/edi/test_maps/map_1.png')

    fraction = 25
    fmp = map_point(140,128,0)
    free_map_points = []
    free_map_points.append(fmp)
    # free_map_points = get_free_map_points(image, fraction=fraction)
    scores = []

    point_laser_origin = [150, 120] 
    point_map_sample = [151, 120] 

    for p in free_map_points:
        m = [p.y +1, p.x]
        l = [p.y, p.x]
        
        print(f"robot pos: (", p.x, ",", p.y,")")

        laser_points = find_nearest_non_white_pixels(image, l, 0, detection_color=100)
        map_sample_points = find_nearest_non_white_pixels(image, m, 0, detection_color=50)

        laser_color = (255, 0, 0)  # Green color in RGB format
        map_sample_color = (0, 252, 0)  # Green color in RGB format

        image = color_nearest_pixels(image, laser_points, laser_color)
        image = color_nearest_pixels(image, map_sample_points, map_sample_color)
        image = color_nearest_pixels(image, [[140, 140]], (50, 50, 150), radius=5,white_only=True)

        distances = nearest_neighbor_distances(laser_points, map_sample_points)
        score = sum(distances)
        scores.append(score)

    # normalized_scores = min_max_normalize(scores)

    # for i in range(len(normalized_scores)):
    #     free_map_points[i].score = normalized_scores[i]

    # for p in free_map_points:
    #     # image[p.y, p.x] = [p.score, p.score, p.score]
    #     image = color_nearest_pixels(image, [[p.y, p.x]], (255, 255-(p.score*255), 0), radius=round(256/fraction), white_only=True)

    display_size = 800
    display_result(image, display_size)







# Using the special variable 
# __name__
if __name__=="__main__":
    main()