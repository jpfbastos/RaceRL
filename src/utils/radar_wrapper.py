import numpy as np
import config
import cv2

CAR_POSITION_YX = (70, 48)

def is_offroad(pixel):
    return pixel[1] > 150 > pixel[2] and pixel[0] < 150

def cast_pixel_ray(world, angle_rad, max_distance=70, step_size=2, display=False):

    for dist in range(0, max_distance, step_size):
        y = CAR_POSITION_YX[0] + int(np.sin(angle_rad) * dist)
        x = CAR_POSITION_YX[1] + int(np.cos(angle_rad) * dist)

        if not (0 <= x < world.shape[1] and 0 <= y < world.shape[0]):
            return dist

        pixel = world[y, x]

        if is_offroad(pixel):
            if display:
                cv2.line(world, (CAR_POSITION_YX[1], CAR_POSITION_YX[0]), (x, y), color=(0, 0, 255), thickness=1)
                cv2.imshow("Game", world)
                cv2.waitKey(1)
            return dist  # Hit something

    if display:
        y = CAR_POSITION_YX[0] + int(np.sin(angle_rad) * max_distance)
        x = CAR_POSITION_YX[1] + int(np.cos(angle_rad) * max_distance)
        cv2.line(world, (CAR_POSITION_YX[1], CAR_POSITION_YX[0]), (x, y), color=(0, 0, 255), thickness=1)
        cv2.imshow("Game", world)
        cv2.waitKey(1)

    return max_distance  # No hit

def get_radar_readings(world, num_rays=5, ray_length=70, max_angle=90, display=False):
    readings = []
    angles = np.linspace(-2*max_angle, 0, num_rays)
    for angle in angles:
        angle = np.deg2rad(angle)
        dist = cast_pixel_ray(world, angle, ray_length, display=display)
        readings.append(dist)
    return np.array(readings)
