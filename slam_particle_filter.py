import numpy as np
import matplotlib.pyplot as plt

# --- Simulation settings ---
np.random.seed(42)
num_particles = 100
true_pos = np.array([5.0, 5.0])
landmark_pos = np.array([8.0, 5.0])
sensor_std = 0.1  # Very accurate sensor

# --- Simulate particles ---
particles = np.random.uniform(low=3, high=7, size=(num_particles, 2))  # Random positions in 2D

# --- Simulate sensor reading from true position ---
true_measurement = np.linalg.norm(true_pos - landmark_pos)

# --- Compute weights based on likelihood ---
def compute_weight(particle):
    expected_measurement = np.linalg.norm(particle - landmark_pos)
    error = true_measurement - expected_measurement
    likelihood = np.exp(-0.5 * (error / sensor_std)**2)
    return likelihood

weights = np.array([compute_weight(p) for p in particles])
weights /= np.sum(weights)  # Normalize

# --- Resample based on weights ---
resampled_indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
resampled_particles = particles[resampled_indices]

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Before resampling
axs[0].scatter(particles[:, 0], particles[:, 1], c=weights, cmap='viridis', s=50)
axs[0].scatter(*true_pos, c='red', marker='*', s=150, label='True Position')
axs[0].scatter(*landmark_pos, c='black', marker='^', s=100, label='Landmark')
axs[0].set_title("Before Resampling (Color = Weight)")
axs[0].legend()
axs[0].axis('equal')

# After resampling
axs[1].scatter(resampled_particles[:, 0], resampled_particles[:, 1], c='blue', s=50)
axs[1].scatter(*true_pos, c='red', marker='*', s=150, label='True Position')
axs[1].scatter(*landmark_pos, c='black', marker='^', s=100, label='Landmark')
axs[1].set_title("After Resampling (Loss of Diversity)")
axs[1].legend()
axs[1].axis('equal')

plt.tight_layout()
plt.show()
