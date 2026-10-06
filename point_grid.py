import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw

class PointGrid(mglw.WindowConfig):
    gl_version = (3,3)
    title = "Point Grid"
    window_size = (512, 512)
    aspect_ratio = 1.0
    resizable = True
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.load_program(
            vertex_shader="point_grid.vert",
            fragment_shader="point_grid.frag",
        )

        x = np.linspace(-1.0, 1.0, 50)
        y = np.linspace(-1.0, 1.0, 50)
        X, Y = np.meshgrid(x, y)
        n = X.size
        colors = np.random.rand(n,3)
        vertices = np.column_stack([X.ravel(),Y.ravel(),colors]).astype("f4")

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, "2f 3f", "in_vert", "in_color")]
        )

        self.ctx.point_size = 4.0
        self.esc_pressed = False
        self.wnd.exit_key = None # To stop default behaviour

    def on_render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.POINTS)

    def on_key_event(self, key, action, modifiers):

        # Logic to close window when escape key is released
        if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_PRESS:
            self.esc_pressed = True
        if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_RELEASE:
            if self.esc_pressed:
                self.esc_pressed = False
                self.wnd.close()

if __name__ == "__main__":
    mglw.run_window_config(PointGrid)