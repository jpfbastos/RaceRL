import gymnasium as gym
import radar_wrapper as radar
import cv2

# Create the CarRacing environment
env = gym.make("CarRacing-v3", render_mode="rgb_array", domain_randomize=False, continuous=False)

# Reset the environment to start
obs, info = env.reset()
car = env.unwrapped.car
print(car)
world = env.unwrapped.world

done = False
count = 0
while not done:
    frame = env.render()
    cv2.imshow("Game", frame)
    cv2.waitKey(1)
    readings = radar.get_radar_readings(frame, 11, 200)
    action = 0
    if (readings[2] - readings[-3]) <= -10:
        action = 1  # turn left
    elif (readings[2] - readings[-3]) > 10:
        action = 2  # turn right
    elif (readings[0] - readings[-1]) <= -70:
        if count % 2 == 0:
            action = 1
    elif (readings[0] - readings[-1]) > 70:
        if count % 2 == 0:
            action = 2
    count += 1
    obs, reward, terminated, truncated, info = env.step(action)
    if readings[5] < 150 and count % (readings[5] // 5) == 0:
        action = 4  # brake
    elif readings[5] > 150 or count % (200 - readings[5] // 5) == 0:
        action = 3  # move
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if truncated:
        print("truncated")
        done = False
    if terminated:
        print("terminated")

# Close the environment properly
env.close()
