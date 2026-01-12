import matplotlib.pyplot as plt
import csv
import time

csv_file = 'training_log.csv'

while True:
    try:
        # Read data
        epochs = []
        rewards = []
        entropies = []
        value_stds = []
        value_means = []
        return_means = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                rewards.append(float(row['reward']))
                entropies.append(float(row['entropy']))
                value_stds.append(float(row['value_std']))
                value_means.append(float(row['value_mean']))
                return_means.append(float(row['return_mean']))

        # Close old figure and create new one each time
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].plot(epochs, rewards)
        axes[0, 0].set_title('Reward')
        axes[0, 0].set_ylim([-100, 100])

        axes[0, 1].plot(epochs, entropies)
        axes[0, 1].set_title('Entropy')

        axes[1, 0].plot(epochs, value_stds)
        axes[1, 0].set_title('Value Std')

        axes[1, 1].plot(epochs, return_means, label='Returns')
        axes[1, 1].plot(epochs, value_means, label='Values')
        axes[1, 1].set_title('Returns vs Values')
        axes[1, 1].set_ylim([-100, 10])
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(30)

    except FileNotFoundError:
        print("Waiting for training_log.csv...")
        time.sleep(2)
    except KeyboardInterrupt:
        break

plt.close('all')