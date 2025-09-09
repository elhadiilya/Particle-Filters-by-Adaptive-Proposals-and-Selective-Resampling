import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# --- Settings ---
np.random.seed(42)
num_particles = 100
map_size = (10, 10)
true_pos = np.array([5.0, 5.0])
landmark_pos = np.array([8.0, 5.0])
sensor_std = 0.1  # Very accurate sensor

# --- Generate initial particles using motion model (uniform around last position) ---
particles = true_pos + np.random.normal(0, 1.0, size=(num_particles, 2))

# --- Simulate true sensor measurement (range to landmark) ---
true_measurement = np.linalg.norm(true_pos - landmark_pos)

# --- Compute sensor likelihood for each particle ---
def sensor_likelihood(particle):
    expected_measurement = np.linalg.norm(particle - landmark_pos)
    error = true_measurement - expected_measurement
    return np.exp(-0.5 * (error / sensor_std)**2)

likelihoods = np.array([sensor_likelihood(p) for p in particles])
likelihoods += 1e-300  # Avoid division by zero
likelihoods /= np.sum(likelihoods)

# --- Sensor-based proposal: Fit Gaussian around high-likelihood region ---
# Weighted mean
mu = np.sum(particles * likelihoods[:, np.newaxis], axis=0)

# Weighted covariance
diffs = particles - mu
cov = np.dot((diffs * likelihoods[:, np.newaxis]).T, diffs)
# --- Resample based on weights ---
resampled_indices = np.random.choice(np.arange(num_particles), size=num_particles, p=likelihoods)
resampled_particles = particles[resampled_indices]

# --- Sample new particles from this Gaussian proposal ---
resampled_particles = np.random.multivariate_normal(mean=mu, cov=cov, size=num_particles)

# --- Plot ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Before sensor-based resampling
axs[0].scatter(particles[:, 0], particles[:, 1], c=likelihoods, cmap='viridis', s=50)
axs[0].scatter(*true_pos, c='red', marker='*', s=150, label='True Position')
axs[0].scatter(*landmark_pos, c='black', marker='^', s=100, label='Landmark')
axs[0].set_title("Before: Particles & Sensor Likelihood")
axs[0].legend()
axs[0].axis('equal')
axs[0].set_xlim(2, 8)
axs[0].set_ylim(2, 8)

# After sensor-based resampling
axs[1].scatter(resampled_particles[:, 0], resampled_particles[:, 1], c='blue', s=50)
axs[1].scatter(*true_pos, c='red', marker='*', s=150, label='True Position')
axs[1].scatter(*landmark_pos, c='black', marker='^', s=100, label='Landmark')
axs[1].set_title("After: Sensor-Based Proposal Sampling")
axs[1].legend()
axs[1].axis('equal')
axs[1].set_xlim(2, 8)
axs[1].set_ylim(2, 8)

plt.tight_layout()
plt.show()
