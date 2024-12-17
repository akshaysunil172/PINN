# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:56:21 2024

@author: AKSHAY SUNIL
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Check if GPU is available and TensorFlow is using it
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class PINN(tf.keras.Model):
    def __init__(self, layers, outputs=1):
        super(PINN, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(neurons, activation=tf.nn.swish) for neurons in layers]
        self.output_layer = tf.keras.layers.Dense(outputs, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

def compute_loss(model, inputs, D, v_x, K, 
                 lambda_pde=200.0, 
                 lambda_left=1000.0, 
                 lambda_right=1000.0, 
                 lambda_top=1000.0, 
                 lambda_bottom=1000.0,
                 lambda_nonlinear=1.0, 
                 lambda_advection=300.0):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        t = inputs[:, 0:1]
        x = inputs[:, 1:2]
        C = model(tf.stack([t[:, 0], x[:, 0]], axis=1))
        C_t = tape.gradient(C, t)
        C_x = tape.gradient(C, x)
        C_xx = tape.gradient(C_x, x)

    D_x = tf.gather(K, tf.cast(x // 1, tf.int32))
    f = C_t + lambda_advection * v_x * C_x - D_x * C_xx
    loss_pde = tf.reduce_mean(tf.square(f))

    # Applying different weights to different boundary conditions
    boundary_loss_left = tf.reduce_mean(tf.square(C[x[:, 0] == 0] - 5))  # Left boundary (C = 5)
    boundary_loss_right_neumann = tf.reduce_mean(tf.square(C_x[x[:, 0] == 50]))  # Neumann at right boundary
    boundary_loss_top = tf.reduce_mean(tf.square(C[t[:, 0] == tf.reduce_max(t)] - 0))  # Top boundary (C = 0)
    boundary_loss_bottom = tf.reduce_mean(tf.square(C[t[:, 0] == 0] - 0))  # Bottom boundary (C = 0)

    loss_nonlinear = tf.reduce_mean(tf.square(C_xx))  # Penalize sharp concentration variations

    # Combining boundary losses with individual weights
    loss_boundary = (lambda_left * boundary_loss_left + 
                     lambda_right * boundary_loss_right_neumann + 
                     lambda_top * boundary_loss_top + 
                     lambda_bottom * boundary_loss_bottom)

    total_loss = lambda_pde * loss_pde + loss_boundary + lambda_nonlinear * loss_nonlinear
    del tape
    return total_loss, loss_pde, loss_boundary, loss_nonlinear

def train_model(model, optimizer, epochs, domain_points, v_x, D, K, lambda_pde_initial=1.0, lambda_boundary_initial=10.0, lambda_nonlinear_initial=1.0, lambda_advection=10.0, decay_rate=0.001):
    loss_history = {"total": [], "pde": [], "boundary": [], "nonlinear": []}
    for epoch in range(epochs):
        lambda_pde = lambda_pde_initial * np.exp(-decay_rate * epoch)
        lambda_boundary = lambda_boundary_initial * np.exp(-decay_rate * epoch)
        lambda_nonlinear = lambda_nonlinear_initial * np.exp(-decay_rate * epoch)

        with tf.GradientTape() as tape:
            loss, loss_pde, loss_boundary, loss_nonlinear = compute_loss(model, domain_points, D, v_x, K, lambda_pde, lambda_boundary, lambda_nonlinear, lambda_advection)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_history["total"].append(loss.numpy())
        loss_history["pde"].append(loss_pde.numpy())
        loss_history["boundary"].append(loss_boundary.numpy())
        loss_history["nonlinear"].append(loss_nonlinear.numpy())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.numpy()}, PDE Loss: {loss_pde.numpy()}, Boundary Loss: {loss_boundary.numpy()}, Nonlinear Loss: {loss_nonlinear.numpy()}")
    return loss_history

# Setup for domain, time, space, and hydraulic conductivity
length, width, dx, dy = 50, 50, 1, 1
nx, ny = int(length / dx) + 1, int(width / dy) + 1
t = np.linspace(0, 50, 100)
x = np.linspace(0, 50, 100)
T, X = np.meshgrid(t, x)
domain_points = np.stack([T.flatten(), X.flatten()], axis=-1)
domain_points = tf.convert_to_tensor(domain_points, dtype=tf.float32)

K = np.zeros(ny)
K[:ny // 3] = 4.5
K[ny // 3: 2 * ny // 3] = 6.0
K[2 * ny // 3:] = 9.0
K = tf.convert_to_tensor(K, dtype=tf.float32)

model = PINN(layers=[512, 512, 512, 512], outputs=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

loss_history = train_model(model, optimizer, 10000, domain_points, 0.1, 0.01, K, lambda_pde_initial=100.0, lambda_boundary_initial=500.0, lambda_nonlinear_initial=1.0, lambda_advection=100.0, decay_rate=0.001)

# Visualization of the results
predicted_concentration = model(domain_points).numpy().reshape(100, 100)

# Analytical Solution Code
Dxx, Dyy = 0.1, 0.1  # Dispersion coefficients
v = 0.1  # Velocity
C_left = 5.0  # Continuous source concentration
dt = 1.0  # Time step
total_time = 300  # Simulation duration

x_analytical = np.linspace(0, length, nx)
y_analytical = np.linspace(0, width, ny)
X_analytical, Y_analytical = np.meshgrid(x_analytical, y_analytical)

# Initialize concentration array
C_analytical = np.zeros((ny, nx))

# Time-stepping loop
for t in range(1, int(total_time / dt) + 1):
    C_analytical[:, 0] = C_left  # Continuous source at left boundary
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            diffusion_x = Dxx * (C_analytical[i, j+1] - 2*C_analytical[i, j] + C_analytical[i, j-1]) / dx**2
            diffusion_y = Dyy * (C_analytical[i+1, j] - 2*C_analytical[i, j] + C_analytical[i-1, j]) / dy**2
            advection = v * (C_analytical[i, j] - C_analytical[i, j-1]) / dx
            C_analytical[i, j] += dt * (diffusion_x + diffusion_y - advection)

# Visualization of the results
plt.figure(figsize=(15, 7))

# PINN Solution Plot
plt.subplot(1, 2, 1)
im1 = plt.imshow(np.transpose(predicted_concentration), extent=[0, 50, 0, 50], origin='lower', cmap='viridis', aspect='auto')
plt.title("PINN Solution after 300 Days")
plt.xlabel("Distance (m)")
plt.ylabel("Width (m)")

# Analytical Solution Plot
plt.subplot(1, 2, 2)
im2 = plt.imshow(C_analytical, extent=[0, 50, 0, 50], origin='lower', cmap='viridis', aspect='auto')
plt.title("Analytical Solution after 300 Days")
plt.xlabel("Distance (m)")
plt.ylabel("Width (m)")

# Common Colorbar
fig = plt.gcf()
#cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03]) # Adjust the position and size of the color bar

cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.03]) # Lower the color bar further by adjusting the second value
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
plt.tight_layout()
plt.show()
