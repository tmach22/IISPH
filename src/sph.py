from turtle import position
from renderer import ParticleRenderer
import numpy as np

GRAVITY = np.array([0.0, -9.81])  # Gravity force
VISCOSITY = 0.01  # Viscosity coefficient

class Particle:
    def __init__(self, position, velocity, mass=1.0):
        self.position = np.array(position, dtype=np.float32) # x, y coordinates
        self.velocity = np.array(velocity, dtype=np.float32) # vx, vy
        self.velocity_star = np.array([0.0, 0.0], dtype=np.float32) # vx, vy
        self.mass = mass
        self.density = 1.0
        self.density_star = 0.0
        self.pressure = 0.0
        self.density_gradient = np.array([0.0, 0.0], dtype=np.float32)  # for visualization
       

class SPHSystem:
    def __init__(self, num_particles, domain_size=1.0, time_step=0.1):
        self.particles = [Particle(position=[np.random.uniform(0.2,0.5), np.random.uniform(0.35,1)], velocity=[0.0, 0.0]) for _ in range(num_particles)]
        self.domain_size = domain_size
        self.particle_spacing = domain_size / np.sqrt(num_particles)
        self.h = np.round(2 * self.particle_spacing, 4)
        self.dt = time_step
        self.boundary_slope = -0.35
        self.boundary_intercept = 0.3
        self.left_wall_x = 0.1
        self.right_wall_x = 0.8

    def print_particles(self):
        for i, p in enumerate(self.particles):
            print(f"Particle {i}: Position={p.position}, Velocity={p.velocity}, Density={p.density}, Pressure={p.pressure} Mass={p.mass}");

    def poly6_kernel(self, r, h):
        if 0 <= r <= h:
            coef = 315/(64 * np.pi * h**9)
            return coef * (h**2 - r**2)**3
        return 0.0

    def compute_density(self):
        for p_i in self.particles:
            density = 0.0
            for p_j in self.particles:
                r = np.linalg.norm(p_i.position - p_j.position)
                density += p_j.mass * self.poly6_kernel(r, self.h)
            p_i.density = density

    def equation_of_state(self, density, rho0=1000, k=3.0):
        return k*(density - rho0)

    def compute_pressure(self):
        for p in self.particles:
            p.pressure = self.equation_of_state(p.density)

    def print_pressures(self):
        for i, p in enumerate(self.particles):
            print(f"Particle {i}: Pressure={p.pressure}")

    def print_densities(self):
        for i, p in enumerate(self.particles):
            print(f"Particle {i}: Density={p.density}")

    def viscosity_kernel_laplacian(self, r, h):
        """Laplacian of the viscosity kernel"""
        if 0 < r <= h:
            return (45/(np.pi * self.h**6)) * (h-r)
    
    def compute_advection_velocity(self):
        """Compute intermediate velocity with gravity + viscosity forces."""
        for p_i in self.particles:
            gravity_force = GRAVITY*p_i.mass # Gravity force
            viscous_force = np.array([0.0, 0.0]) # Initialize viscous force

            for p_j in self.particles:
                if p_i is not p_j:
                    r = np.linalg.norm(p_i.position - p_j.position)
                    if r > 0 and r < self.h:
                        laplacian_w = self.viscosity_kernel_laplacian(r, self.h)
                        viscous_force += VISCOSITY * p_j.mass * ((p_j.velocity - p_i.velocity) / p_j.density) * laplacian_w

            total_force = gravity_force + viscous_force
            p_i.velocity_star = p_i.velocity + (self.dt * total_force/ p_i.density)

    def print_advection_velocities(self):
        """Print advection velocities for testing."""
        for i, p in enumerate(self.particles):
            print(f"Particles {i}: Advection Velocity={p.velocity_star}")

    def spiky_kernel_gradient(self, r_vec, h):
        """Gradient of the spiky kernel for pressure computation."""
        r = np.linalg.norm(r_vec)
        if 0 < r <= h:
            coef = -45 / (np.pi * h**6)
            return coef * (h - r)**2 * (r_vec / r)
        return np.array([0.0, 0.0])

    def compute_advection_density(self):
        """Predict densities using advection velocities."""
        for p_i in self.particles:
            rho_star = p_i.density # Start with the current density
            for p_j in self.particles:
                if p_i is not p_j:
                    r_vec = p_i.position - p_j.position
                    r = np.linalg.norm(p_i.position - p_j.position)
                    if r > 0 and r < self.h:
                        grad_w = self.spiky_kernel_gradient(r_vec, self.h)
                        rho_star += self.dt * p_j.mass * (p_i.velocity_star - p_j.velocity_star).dot(grad_w)
            p_i.density_star = rho_star # store predicted density

    def print_predicted_densities(self):
        """Print predicted densities for testing."""
        for i, p in enumerate(self.particles):
            print(f"Particle {i}: Predicted Density={p.density_star}")

    def compute_coeff_matrix(self):
        """Compute coefficient matrix A for the PPE equation"""
        N = len(self.particles)
        A = np.zeros((N, N))
        b = np.zeros(N)

        for i, p_i in enumerate(self.particles):
            sum_aij = 0
            bi = p_i.density - p_i.density_star # Right-hand side (density error)

            for j, p_j in enumerate(self.particles):
                if i != j:
                    r = np.linalg.norm(p_i.position - p_j.position)
                    if r > 0 and r < self.h:
                        grad_w = self.spiky_kernel_gradient(r, self.h)
                        aij = (p_j.mass/p_j.density) * np.dot(grad_w, grad_w)
                        A[i, j] = -aij # off diagonal elements
                        sum_aij += aij
            A[i,i] = sum_aij # Diagonal elements
            b[i] = bi # Right-hand side

        # Diagonal preconditioning: Scale A and b by diagonal entries
        for i in range(N):
            diag_val = A[i, i]
            if diag_val != 0:
                A[i, :] /= diag_val  # Scale row i by 1/A[i,i]
                b[i] /= diag_val

        return A, b

    def relaxed_jacobi_solve(self, A, b, max_iterations=100, omega=0.5, epsilon=1e-4):
        """Solve PPE using Relaxed Jacobi Iteration."""
        N = len(b)
        P = np.zeros(N) # Intialize pressures to zero

        for iteration in range(max_iterations):
            P_new = np.copy(P)
            residual_new = 0

            for i in range(N):
                sigma = np.dot(A[i, :], P) - A[i, i] * P[i]  # A[i,i] = 1 after scaling
                P_new[i] = (1 - omega) * P[i] + omega * (b[i] - sigma)  # No division by A[i,i]
                residual_new += abs(P_new[i] - P[i])
            P = P_new
            if np.max(np.abs(residual_new)) < epsilon:
                break
        return P

    def solve_pressure(self):
        """Compute pressure using the PPE and Relaxed Jacobi with convergence."""
        A, b =  self.compute_coeff_matrix()
        pressure = self.relaxed_jacobi_solve(A, b)
        for i, p in enumerate(self.particles):
            p.pressure = pressure[i]

    def apply_pressure_forces(self):
        """Correct velocities by applying pressure forces using computed pressures.
        v_i^{n+1} = v_i^* - dt * sum_j ( m_j (p_i/ρ_i^2 + p_j/ρ_j^2) * grad W_ij )
        """
        for p_i in self.particles:
            pressure_force = np.array([0.0, 0.0], dtype=np.float32)
            for p_j in self.particles:
                if p_i is not p_j:
                    r = np.linalg.norm(p_i.position - p_j.position)
                    if r > 0 and r < self.h:
                        grad_w = self.spiky_kernel_gradient(p_i.position-p_j.position, self.h)
                        pressure_term = (p_i.pressure/(p_i.density**2)) + (p_j.pressure/(p_j.density**2))
                        pressure_force += p_j.mass * pressure_term * grad_w
            p_i.velocity = p_i.velocity_star - self.dt * pressure_force

    def print_velocities(self):
        for i, p in enumerate(self.particles):
            print(f"Particle {i}: Corrected Velocity = {p.velocity}")

    def compute_density_gradient(self):
        """Compute density gradient for visualization using a simple SPH gradient approximation."""
        for p_i in self.particles:
            grad_rho = np.array([0.0, 0.0], dtype=np.float32)
            for p_j in self.particles:
                if p_i is not p_j:
                    r_vec = p_i.position - p_j.position
                    r = np.linalg.norm(r_vec)
                    if 0 < r < self.h:
                        grad_w = self.spiky_kernel_gradient(r_vec, self.h)
                        grad_rho += p_j.mass * (p_j.density - p_i.density) * grad_w
            p_i.density_gradient = grad_rho

if __name__ == "__main__":
    sph = SPHSystem(num_particles=40)
    sph.compute_density()
    renderer = ParticleRenderer(sph)
    renderer.run()