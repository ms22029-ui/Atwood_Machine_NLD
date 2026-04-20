import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- System Parameters ---
g = 9.81

# --- Equations of Motion ---
def sam_derivatives(t, y, mu):
    r, vr, theta, vtheta = y
    
    if r < 1e-5:
        r = 1e-5
        
    r_dot = vr
    vr_dot = (r * vtheta**2 - g * (mu - np.cos(theta))) / (mu + 1)
    theta_dot = vtheta
    vtheta_dot = -(2 * vr * vtheta + g * np.sin(theta)) / r
    
    return [r_dot, vr_dot, theta_dot, vtheta_dot]

# --- Event Detection (The Poincaré Slice) ---
def cross_theta_zero(t, y, mu):
    return y[2] 

cross_theta_zero.direction = 1

# --- Optimized Simulation Setup ---
t_span = (0, 8000)  # Reduced from 15000 for faster execution
y0 = [1.0, 0.0, np.pi/2, 0.0]

print("Generating Poincaré section for mu = 4...")
sol_mu4 = solve_ivp(sam_derivatives, t_span, y0, args=(4,), 
                    method='RK45', rtol=1e-6, atol=1e-6,  # Loosened tolerances
                    events=cross_theta_zero)

print("Generating Poincaré section for mu = 3...")
sol_mu3 = solve_ivp(sam_derivatives, t_span, y0, args=(3,), 
                    method='RK45', rtol=1e-6, atol=1e-6,  # Loosened tolerances
                    events=cross_theta_zero)

# --- Extracting the Event Points ---
poincare_mu4_r = sol_mu4.y_events[0][:, 0]
poincare_mu4_vr = sol_mu4.y_events[0][:, 1]

poincare_mu3_r = sol_mu3.y_events[0][:, 0]
poincare_mu3_vr = sol_mu3.y_events[0][:, 1]

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plt.style.use('default')

# mu = 4: Regular Motion
axs[0].scatter(poincare_mu4_r, poincare_mu4_vr, s=2, color='navy', alpha=0.7)
axs[0].set_title(r'Poincaré Section ($\mu=4$): Regular Motion')
axs[0].set_xlabel(r'$r$ (Radial Position)')
axs[0].set_ylabel(r'$v_r$ (Radial Velocity)')
axs[0].grid(True, alpha=0.3)

# mu = 3: Chaotic Sea
axs[1].scatter(poincare_mu3_r, poincare_mu3_vr, s=2, color='darkred', alpha=0.7)
axs[1].set_title(r'Poincaré Section ($\mu=3$): Chaotic Sea')
axs[1].set_xlabel(r'$r$ (Radial Position)')
axs[1].set_ylabel(r'$v_r$ (Radial Velocity)')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
