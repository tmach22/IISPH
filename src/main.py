import glfw
from OpenGL.GL import *

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Create a window
    window = glfw.create_window(800, 600, "OpenGL Test", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    # Main loop
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
