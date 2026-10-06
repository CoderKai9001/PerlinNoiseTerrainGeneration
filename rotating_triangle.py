import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw

def perspective(fovy_deg, aspect, near, far):
    f = 1.0/np.tan(np.radians(fovy_deg)/2.0)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
        [0,0,-1,0]
    ], dtype="f4")

def translate(x,y,z):
    m = np.eye(4, dtype="f4")
    m[:3,3] = [x,y,z]
    return m

def rotate_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype="f4")

class Triangle(mglw.WindowConfig):
    gl_version = (3,3)
    title = "Point Grid"
    window_size = (512, 512)
    aspect_ratio = 2.0
    resizable = True
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.load_program(
            vertex_shader="triangle.vert",
            fragment_shader="triangle.frag"
        )

        # x,y,r,g,b
        vertices = np.array([
            0.0, 0.6, 0.0, 1.0, 0.0, 0.0,
            0.6, -0.6, 0.0, 0.0, 0.0, 1.0,
            -0.6, -0.6, 0.0, 0.0, 1.0, 0.0,
        ], dtype="f4")

        self.tri_vbo = self.ctx.buffer(vertices.tobytes())
        self.tri_vao = self.ctx.vertex_array(self.program, [(self.tri_vbo, "3f 3f", "coord", "in_color")])

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2,0.2,0.2,1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = perspective(60.0, self.wnd.aspect_ratio, 0.1, 100.0)
        view = translate(0.0, 0.0, -3.0)
        model = rotate_y(time)

        mvp = proj @ view @ model
        self.program["u_mvp"].write(mvp.T.tobytes())
        self.tri_vao.render(moderngl.TRIANGLES)

if __name__ == "__main__":
    mglw.run_window_config(Triangle)
