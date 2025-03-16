import time
import glfw
from OpenGL.GL import *
import numpy as np

class ParticleRenderer:
    def __init__(self, sph_system, width=800, height=600):
        self.sph_system = sph_system
        self.width = width
        self.height = height

        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW could not be initialised!")

        # Create Window
        self.window = glfw.create_window(self.width, self.height, "SPH Visualization", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW Window could not be created!")

        glfw.make_context_current(self.window)

    def draw_boundary(self):
        """Draw the sloping boundary line: y = m * x + b."""
        m = self.sph_system.boundary_slope
        b_val = self.sph_system.boundary_intercept
        left_wall_x = self.sph_system.left_wall_x  # Get wall positions
        right_wall_x = self.sph_system.right_wall_x

        left_intersect_y = m * left_wall_x + b_val
        right_intersect_y = m * right_wall_x + b_val

        glLineWidth(2)
        glColor3f(1.0, 1.0, 1.0)  # White line for boundary
        glBegin(GL_LINES)

        x0 = max(0.0, left_wall_x)  # Start at left wall or 0
        y0 = m * x0 + b_val
        x1 = min(1.0, right_wall_x)  # End at right wall or 1
        y1 = m * x1 + b_val

        # Convert [0,1] domain to OpenGL normalized [-1, 1]
        glVertex2f(x0 * 2 - 1, y0 * 2 - 1)
        glVertex2f(x1 * 2 - 1, y1 * 2 - 1)
        glEnd()

        # Draw the vertical walls
        glBegin(GL_LINES)
        # Left wall
        y0 = max(-1.0, left_intersect_y)
        y1 = 1.0
        glVertex2f(left_wall_x * 2 - 1, y0 * 2 - 1)
        glVertex2f(left_wall_x * 2 - 1, y1 * 2 - 1)
        # Right wall
        y0 = max(-1.0, right_intersect_y)
        y1 = 1.0
        glVertex2f(right_wall_x * 2 - 1, y0 * 2 - 1)
        glVertex2f(right_wall_x * 2 - 1, y1 * 2 - 1)
        glEnd()

    def draw_particles(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        glPointSize(5) # particle size
        glBegin(GL_POINTS)

        for p in self.sph_system.particles:
            x, y = p.position
            x = (x*2)-1 # normalise for OpenGL (-1 to 1 range)
            y = (y*2)-1 
            pressures = [pp.pressure for pp in self.sph_system.particles]
            minP, maxP = min(pressures), max(pressures)
            if maxP - minP < 1e-6:
                t = 0.5
            else:
                t = (p.pressure - minP) / (maxP - minP)
            # Color: Blue (low pressure) to Red (high pressure)
            r_color = t
            b_color = 1 - t
            glColor3f(0.0, 0.5, 1.0) # Set particle color (blue)
            glVertex2f(x, y)

        glEnd()


    def update_positions(self):
        """Update particle positions using the corrected velocity and apply boundary conditions for a sloping surface."""
        self.sph_system.compute_advection_velocity()
        self.sph_system.compute_advection_density()
        self.sph_system.solve_pressure()
        self.sph_system.apply_pressure_forces()
        m = self.sph_system.boundary_slope
        b_val = self.sph_system.boundary_intercept
        left_wall_x = self.sph_system.left_wall_x
        right_wall_x = self.sph_system.right_wall_x

        # Compute the normalized normal vector of the boundary.
        # For a line y = m*x + b, a tangent is (1, m), so a normal is (-m, 1)
        n = np.array([-m, 1.0])
        n = n / np.linalg.norm(n)

        for p in self.sph_system.particles:
            p.position += self.sph_system.dt * p.velocity
            # Compute the y-value of the sloping boundary at particle's x position
            boundary_y = m * p.position[0] + b_val
            # If particle goes below the boundary, reposition it and reflect its y-velocity
            if p.position[1] < boundary_y:
                p.position[1] = boundary_y
                # Decompose the velocity into normal and tangential components.
                v = p.velocity
                v_n = np.dot(v, n) * n       # Normal component.
                v_t = v - v_n                # Tangential component.

                # Reflect the normal component with a restitution coefficient.
                restitution = 0.3         # Adjust as needed.
                v_n_reflected = -restitution * v_n

                # Apply friction to the tangential component.
                friction = 0.99              # Adjust as needed.
                v_t *= friction

                # Combine the new normal and tangential components.
                p.velocity = v_n_reflected + v_t

            if p.position[0] < left_wall_x:
                p.position[0] = left_wall_x
                p.velocity[0] *= -0.8  # Invert x-velocity (with damping)
            elif p.position[0] > right_wall_x:
                p.position[0] = right_wall_x
                p.velocity[0] *= -0.8  # Invert x-velocity (with damping)

    def run(self):
        while not glfw.window_should_close(self.window):

            glClear(GL_COLOR_BUFFER_BIT)
            self.update_positions()
            self.draw_particles()
            self.draw_boundary()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()