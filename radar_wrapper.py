import numpy as np
import config
import cv2

CAR_POSITION_YX = (70, 48)

def is_offroad(pixel):
    return pixel[1] > 150 > pixel[2] and pixel[0] < 150

def cast_pixel_ray(world, angle_rad, max_distance=70, step_size=2):

    for dist in range(0, max_distance, step_size):
        y = CAR_POSITION_YX[0] + int(np.sin(angle_rad) * dist)
        x = CAR_POSITION_YX[1] + int(np.cos(angle_rad) * dist)

        if not (0 <= x < world.shape[1] and 0 <= y < world.shape[0]):
            return dist

        pixel = world[y, x]

        if is_offroad(pixel):
            # cv2.line(world, (CAR_POSITION_YX[1], CAR_POSITION_YX[0]), (x, y), color=(0, 0, 255), thickness=2)
            # cv2.imshow("Game", world)
            # cv2.waitKey(1)
            return dist  # Hit something

    #y = CAR_POSITION_YX[0] + int(np.sin(angle_rad) * max_distance)
    #x = CAR_POSITION_YX[1] + int(np.cos(angle_rad) * max_distance)
    #cv2.line(world, (CAR_POSITION_YX[1], CAR_POSITION_YX[0]), (x, y), color=(0, 0, 255), thickness=2)
    #cv2.imshow("Game", world)
    #cv2.waitKey(1)

    return max_distance  # No hit

def get_radar_readings(world, num_rays=5, ray_length=70, max_angle=90):
    readings = []
    angles = np.linspace(-2*max_angle, 0, num_rays)
    for angle in angles:
        angle = np.deg2rad(angle)
        dist = cast_pixel_ray(world, angle, ray_length)
        readings.append(dist)
    return np.array(readings)

"""from Box2D import b2RayCastCallback, b2Vec2
import numpy as np
import config
from config import MAX_RADAR_DISTANCE


class RadarRayCallback(b2RayCastCallback):
    def __init__(self, ignore_body=None):
        super().__init__()
        self.ignore_body = ignore_body    # ① ignore self-hits
        self.closest_fraction = 1.0
        self.point = None

    # called for every fixture intersecting the ray
    def ReportFixture(self, fixture, point, normal, fraction):
        # ① skip the car itself
        if fixture.body is self.ignore_body:
            print("car")
            return 1.0          # keep checking others

        # ② keep the closest hit instead of clipping early
        if fraction < self.closest_fraction:
            self.closest_fraction = fraction
            self.point = point

        return 1.0              # continue; don't clip yet


def cast_radar_ray(car, world, angle_offset_deg=0.0,
                   ray_length=config.MAX_RADAR_DISTANCE):

    start = car.hull.position
    angle = car.hull.angle + np.deg2rad(angle_offset_deg)
    end   = start + ray_length * b2Vec2(np.cos(angle), np.sin(angle))

    cb = RadarRayCallback(ignore_body=car.hull)   # skip self
    world.RayCast(cb, start, end)

    if cb.point is not None:                      # we hit something
        distance = ray_length * cb.closest_fraction
    else:                                         # no hit
        distance = ray_length
    return distance


def get_radar_readings(car, world, num_rays=5, ray_length=MAX_RADAR_DISTANCE, max_angle=90):
    readings = []
    angles = np.linspace(-max_angle, max_angle, num_rays)
    for angle in angles:
        print("angle: ", angle)
        dist = cast_radar_ray(car, world, angle, ray_length)
        readings.append(dist)
    return readings


"""
"""
Extends the base CarRacing-v2 environment:
Adds radar functionality:
Cast N rays from the car at specified angles.
For each ray, use PyBox2D raycasting to detect the first fixture hit.
Record distance or normalized distance to that point.
Optionally includes:
Car's speed
Steering angle
Lateral velocity (if you want)
Replaces the original image observation with your custom radar vector.
Still returns reward and done flag as usual.
"""
