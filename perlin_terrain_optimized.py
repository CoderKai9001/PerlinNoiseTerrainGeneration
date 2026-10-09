import sys
import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw
import glm
import hydra
from omegaconf import DictConfig

class GridMesh(mglw.WindowConfig):
    gl_version = (3,3)
    title = "Perlin Terrain"
    window_size = (512, 512)
    aspect_ratio = 1.0
    resizable = True
    resource_dir = (Path(__file__).parent / "shaders").resolve()
    cfg: DictConfig = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.program = self.load_program(
            vertex_shader="terrain_opt.vert",
            fragment_shader="terrain_opt.frag"
        )

        cfg = self.cfg
        t = cfg.terrain

        self.grid_size = t.grid_size
        x = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        z = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        X, Z = np.meshgrid(x,z)
        Y = np.zeros_like(X) # heights and normals are computed in the vertex shader

        terrain_color = tuple(c / 255.0 for c in t.color)

        positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype("f4")

        idxs = np.arange(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size)
        tl = idxs[:-1, :-1] # -> top left corners of each 2x2 cell which we want to divide into triangles
        tr = idxs[:-1, 1:] # -> top right corners of each 2x2 cell which we want to divide into triangles
        bl = idxs[1:, :-1] # -> bottom left corners of each 2x2 cell which we want to divide into triangles
        br = idxs[1:, 1:] # -> bottom right corners of each 2x2 cell which we want to divide into triangles

        triangles = np.stack([tl, br, tr, 
                              tl, bl, br], axis=-1) # Split a 2x2 cell into 2 triangles (6 of [n-1 , n-1] to [n-1, n-1, 6])

        indices = triangles.astype("i4").ravel()
        self.mesh_vbo = self.ctx.buffer(positions.tobytes())
        self.mesh_ibo = self.ctx.buffer(indices.tobytes())
        self.mesh_vao = self.ctx.vertex_array(self.program, 
                                              [(self.mesh_vbo, "3f", "in_pos")], 
                                              index_buffer = self.mesh_ibo, 
                                              index_element_size = 4)   

        # Set uniform variables for the fragment shader for lighting
        light = cfg.lighting
        self.program["u_light_pos"].value     = tuple(light.light_pos)
        self.program["u_light_color"].value   = tuple(light.light_color)
        self.program["u_ambient"].value       = light.ambient
        self.program["u_spec_strength"].value = light.spec_strength
        self.program["u_shininess"].value     = light.shininess
        self.program["u_terrain_color"].value = terrain_color
        self.program["u_cavity_strength"].value = light.cavity_strength

        # Set uniform variables for the vertex shader's fBm noise
        self.program["u_seed"].value          = cfg.seed
        self.program["u_num_octaves"].value   = t.num_octaves
        self.program["u_base_cells"].value    = t.base_cells
        self.program["u_persistence"].value   = t.persistence
        self.program["u_lacunarity"].value    = t.lacunarity
        self.program["u_amplitude"].value     = t.amplitude
        self.program["u_cavity_octaves"].value = light.cavity_octaves

    def on_render(self, time, frame_time):
        self.ctx.clear(0.1, 0.1, 0.1, 1.0, depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Projection variables
        eye = glm.vec3(*self.cfg.camera.eye)
        proj = glm.perspective(glm.radians(self.cfg.camera.fov), self.wnd.aspect_ratio, 0.1, 100.0)
        view = glm.lookAt(eye, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # model = glm.rotate(time, glm.vec3(0.0,1.0,0.0))
        model = glm.mat4(1.0)
        normal_mat = glm.transpose(glm.inverse(glm.mat3(model)))

        # Uniforms to the vertex shader
        self.program["u_mvp"].write(proj*view*model)
        self.program["u_model"].write(model)
        self.program["u_normal_mat"].write(normal_mat)
        self.program["u_view_pos"].write(eye)

        self.ctx.wireframe = self.cfg.wireframe
        self.mesh_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False

@hydra.main(version_base=None, config_path="configs", config_name="perlin_terrain")
def main(cfg: DictConfig):
    GridMesh.cfg = cfg
    sys.argv = sys.argv[:1]
    mglw.run_window_config(GridMesh)

if __name__ == "__main__":
    main()
