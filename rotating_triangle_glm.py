import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw
import glm

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
        
        proj = glm.perspective(glm.radians(60), self.wnd.aspect_ratio, 0.1, 100.0)
        view = glm.lookAt(glm.vec3(0,0,3), glm.vec3(0,0,0), glm.vec3(0,1,0))
        model = glm.rotate(time, glm.vec3(0,1,0))

        self.program["u_mvp"].write(proj*view*model)
        # self.ctx.wireframe = True
        self.tri_vao.render(moderngl.TRIANGLES)

if __name__ == "__main__":
    mglw.run_window_config(Triangle)
