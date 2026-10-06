import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw
import glm

class GridMesh(mglw.WindowConfig):
    gl_version = (3,3)
    title = "Wireframe Mesh"
    window_size = (512, 512)
    aspect_ratio = 1.0
    resizable = True
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.load_program(
            vertex_shader="triangle.vert",
            fragment_shader="triangle.frag"
        )

        self.grid_size = 50

        x = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        z = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        X, Z = np.meshgrid(x,z)
        Y = np.zeros_like(X).astype("f4")
        R = np.ones_like(X).astype("f4")
        G = np.ones_like(X).astype("f4")
        B = np.ones_like(X).astype("f4")
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), R.ravel(), G.ravel(), B.ravel()]).astype("f4")

        idxs = np.arange(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size)
        tl = idxs[:-1, :-1] # -> top left corners of each 2x2 cell which we want to divide into triangles
        tr = idxs[:-1, 1:] # -> top right corners of each 2x2 cell which we want to divide into triangles
        bl = idxs[1:, :-1] # -> bottom left corners of each 2x2 cell which we want to divide into triangles
        br = idxs[1:, 1:] # -> bottom right corners of each 2x2 cell which we want to divide into triangles

        triangles = np.stack([tl, br, tr, 
                              tl, bl, br], axis=-1) # Split a 2x2 cell into 2 triangles

        indices = triangles.astype("i4").ravel()
        self.mesh_vbo = self.ctx.buffer(vertices.tobytes())
        self.mesh_ibo = self.ctx.buffer(indices.tobytes())
        self.mesh_vao = self.ctx.vertex_array(self.program, 
                                              [(self.mesh_vbo, "3f 3f", "coord", "in_color")], 
                                              index_buffer = self.mesh_ibo, 
                                              index_element_size = 4)

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        proj = glm.perspective(glm.radians(60), self.wnd.aspect_ratio, 0.1, 100.0)
        view = glm.lookAt(glm.vec3(1.5,1.5,2.5), glm.vec3(0,0,0), glm.vec3(0,1,0))

        self.program["u_mvp"].write(proj*view)
        self.ctx.wireframe = True
        self.mesh_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False

if __name__ == "__main__":
    mglw.run_window_config(GridMesh)
