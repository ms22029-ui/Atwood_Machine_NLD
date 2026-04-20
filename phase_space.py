import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- System Parameters ---
g = 9.81  # Acceleration due to gravity

# --- Equations of Motion ---
def sam_derivatives(t, y, mu):
    """
    Computes the derivatives for the Swinging Atwood's Machine.
    State vector y = [r, v_r, theta, v_theta]
    """
    r, vr, theta, vtheta = y

    # Mathematical safeguard: prevent the string from fully retracting (r=0 singularity)
    if r < 1e-5:
        r = 1e-5

    r_dot = vr
    vr_dot = (r * vtheta**2 - g * (mu - np.cos(theta))) / (mu + 1)

    theta_dot = vtheta
    vtheta_dot = -(2 * vr * vtheta + g * np.sin(theta)) / r

    return [r_dot, vr_dot, theta_dot, vtheta_dot]

# --- Simulation Setup ---
t_span = (0, 100)  # Simulate for 100 seconds
t_eval = np.linspace(t_span[0], t_span[1], 15000)  # High resolution for smooth curves

# Initial conditions: [r0, vr0, theta0, vtheta0]
# Starting the pendulum at a 90-degree angle from rest
y0 = [1.0, 0.0, np.pi/2, 0.0]

# --- Numerical Integration ---
# We use strict tolerances (rtol, atol) to ensure energy conservation
print("Integrating for mu = 4 (Regular)...")
sol_mu4 = solve_ivp(sam_derivatives, t_span, y0, args=(4,),
                    t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)

print("Integrating for mu = 3 (Chaotic)...")
sol_mu3 = solve_ivp(sam_derivatives, t_span, y0, args=(3,),
                    t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)

# --- Plotting ---
# Set up a 2x2 grid for the plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.style.use('default')

# mu = 4: Radial Phase Space
axs[0, 0].plot(sol_mu4.y[0], sol_mu4.y[1], color='navy', linewidth=0.6)
axs[0, 0].set_title(r'Radial Phase Space ($\mu=4$)')
axs[0, 0].set_xlabel(r'$r$ (Position)')
axs[0, 0].set_ylabel(r'$v_r$ (Velocity)')
axs[0, 0].grid(True, alpha=0.3)

# mu = 4: Angular Phase Space
axs[0, 1].plot(sol_mu4.y[2], sol_mu4.y[3], color='navy', linewidth=0.6)
axs[0, 1].set_title(r'Angular Phase Space ($\mu=4$)')
axs[0, 1].set_xlabel(r'$\theta$ (Angle)')
axs[0, 1].set_ylabel(r'$v_\theta$ (Angular Velocity)')
axs[0, 1].grid(True, alpha=0.3)

# mu = 3: Radial Phase Space
axs[1, 0].plot(sol_mu3.y[0], sol_mu3.y[1], color='darkred', linewidth=0.6)
axs[1, 0].set_title(r'Radial Phase Space ($\mu=3$) - Onset of Chaos')
axs[1, 0].set_xlabel(r'$r$ (Position)')
axs[1, 0].set_ylabel(r'$v_r$ (Velocity)')
axs[1, 0].grid(True, alpha=0.3)

# mu = 3: Angular Phase Space
axs[1, 1].plot(sol_mu3.y[2], sol_mu3.y[3], color='darkred', linewidth=0.6)
axs[1, 1].set_title(r'Angular Phase Space ($\mu=3$) - Onset of Chaos')
axs[1, 1].set_xlabel(r'$\theta$ (Angle)')
axs[1, 1].set_ylabel(r'$v_\theta$ (Angular Velocity)')
axs[1, 1].grid(True, alpha=0.3)

# --- Output ---
plt.tight_layout()
# This saves the file exactly as named in your LaTeX document
plt.savefig('phase_space_placeholder.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'phase_space_placeholder.png'")
plt.show()
