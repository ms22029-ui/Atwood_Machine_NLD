import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
g = 9.81

# --- Equations of Motion ---
def sam_derivatives(y, mu):
    r, vr, theta, vtheta = y
    
    if r < 1e-5:
        r = 1e-5
        
    r_dot = vr
    vr_dot = (r * vtheta**2 - g * (mu - np.cos(theta))) / (mu + 1)
    theta_dot = vtheta
    vtheta_dot = -(2 * vr * vtheta + g * np.sin(theta)) / r
    
    return np.array([r_dot, vr_dot, theta_dot, vtheta_dot])

# --- Custom RK4 Integrator ---
def rk4_step(y, h, mu):
    k1 = sam_derivatives(y, mu)
    k2 = sam_derivatives(y + 0.5 * h * k1, mu)
    k3 = sam_derivatives(y + 0.5 * h * k2, mu)
    k4 = sam_derivatives(y + h * k3, mu)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- Benettin's Algorithm for Lyapunov Exponents ---
def calculate_lyapunov(mu, t_max, h, tau_steps):
    y = np.array([1.0, 0.0, np.pi/2, 0.0]) # Reference trajectory
    
    # Perturbed trajectory (separated by a microscopic distance d0)
    d0 = 1e-8
    y_pert = y + np.array([d0, 0.0, 0.0, 0.0]) 
    
    time = 0.0
    times = []
    lyap_vals = []
    sum_log = 0.0
    
    n_renorms = int(t_max / (h * tau_steps))
    
    for _ in range(n_renorms):
        # Evolve both trajectories for 'tau_steps'
        for _ in range(tau_steps):
            y = rk4_step(y, h, mu)
            y_pert = rk4_step(y_pert, h, mu)
            time += h
            
        # Measure divergence
        delta = y_pert - y
        d = np.linalg.norm(delta)
        
        # Accumulate the log of the distance ratio
        sum_log += np.log(d / d0)
        
        # Record the current estimate of the Maximal Lyapunov Exponent
        times.append(time)
        lyap_vals.append(sum_log / time)
        
        # Renormalize the perturbed trajectory back to distance d0 along the same direction
        y_pert = y + (delta / d) * d0
        
    return times, lyap_vals

# --- Optimized Simulation Setup ---
t_max = 250       # Reduced from 500 to cut execution time in half
h = 0.01          # Integration time step
tau_steps = 10    # Renormalize every 10 steps (every 0.1 seconds)

print("Calculating Lyapunov Exponent for mu = 4...")
times_mu4, lyap_mu4 = calculate_lyapunov(4.0, t_max, h, tau_steps)

print("Calculating Lyapunov Exponent for mu = 3...")
times_mu3, lyap_mu3 = calculate_lyapunov(3.0, t_max, h, tau_steps)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.style.use('default')

# Plot both lines
plt.plot(times_mu4, lyap_mu4, color='navy', label=r'$\mu=4$ (Regular)')
plt.plot(times_mu3, lyap_mu3, color='darkred', label=r'$\mu=3$ (Chaotic)')

# Add a dashed line at zero to visually separate chaotic from regular regimes
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.title('Maximal Lyapunov Exponent (MLE) Convergence')
plt.xlabel('Time (s)')
plt.ylabel(r'$\lambda_{\text{max}}$')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- Output ---
plt.tight_layout()
plt.show() # Displays directly without saving to disk
