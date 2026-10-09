import sys
import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw
import glm
import hydra
from omegaconf import DictConfig

rng = np.random.default_rng(seed=42)

def get_perlin_grid(rows, cols):
    theta = rng.uniform(0.0, 2.0*np.pi, size=(rows, cols))
    grads = np.stack([np.cos(theta), np.sin(theta)], axis=-1) # (rows, cols, 2)

    return grads 

def fade_curve(x): return x*x*x*(x*(6.0*x - 15.0) + 10.0) # 6x^5 - 15x^4 + 10x^3 in Horner form

def lerp(p,q,t):
    return p + t*(q-p)

def get_perlin_noise(X, Z, cells, grads, offset=(0.0, 0.0)):
    
    # Convert the evaluation grid (terrain grid) from [-1,+1] to [0, dim-1]
    px = (X + 1.0) / 2.0 * cells + offset[0]
    pz = (Z + 1.0) / 2.0 * cells + offset[1]

    # Limit the floor computation to inner grid as there cant be 4 neighbouring lattice points for a mesh point on the border
    cx = np.floor(px).astype(int)
    cz = np.floor(pz).astype(int)

    fx = px - cx
    fz = pz - cz

    g00 = grads[cz, cx]
    g10 = grads[cz, cx+1]
    g01 = grads[cz+1, cx]
    g11 = grads[cz+1, cx+1]

    n00 = g00[..., 0] * fx       + g00[..., 1] * fz
    n10 = g10[..., 0] * (fx - 1) + g10[..., 1] * fz
    n01 = g01[..., 0] * fx       + g01[..., 1] * (fz - 1)
    n11 = g11[..., 0] * (fx - 1) + g11[..., 1] * (fz - 1)

    sx = fade_curve(fx)
    sz = fade_curve(fz)
    return lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sz)

def fbm(X, Z, base_cells, num_octaves, persistence=0.5, lacunarity=2.0):
    noise = 0.0
    total_amp = 0.0
    for k in range(num_octaves):
        amp = persistence ** k
        cells = base_cells * lacunarity ** k
        offset = rng.uniform(0.0, 1.0, size=2)   # random shift per octave
        dim = int(np.ceil(cells)) + 2
        grads = get_perlin_grid(dim, dim)
        noise += amp * get_perlin_noise(X, Z, cells, grads, offset)
        total_amp += amp
    return noise / total_amp

def compute_normals(triangles, positions):
    # triangles: (F, 3) vertex indices, positions: (V, 3)
    p0 = positions[triangles[:, 0]]
    p1 = positions[triangles[:, 1]]
    p2 = positions[triangles[:, 2]]

    # Face normals, unnormalized so larger triangles get more weight
    face_n = np.cross(p1 - p0, p2 - p0)            # (F, 3)

    # Accumulate onto vertices; bincount sums repeated indices (much faster than np.add.at)
    idx = triangles.ravel()
    normals = np.stack([np.bincount(idx, weights=np.repeat(face_n[:, c], 3), minlength=len(positions))
                        for c in range(3)], axis=-1)

    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals.astype("f4")

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
            vertex_shader="terrain.vert",
            fragment_shader="terrain.frag"
        )

        global rng
        cfg = self.cfg
        t = cfg.terrain
        rng = np.random.default_rng(seed=cfg.seed)

        self.grid_size = t.grid_size
        x = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        z = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        X, Z = np.meshgrid(x,z)
        # The noise is evaluated on a row of x and a column of z; broadcasting expands it to the full
        # grid, so the per-point floor/fade work only runs on grid_size values instead of grid_size^2
        Y = (t.amplitude * fbm(x[None, :], z[:, None], t.base_cells, t.num_octaves, t.persistence, t.lacunarity)).astype("f4")
        R = np.full_like(X, t.color[0]/255.0).astype("f4")
        G = np.full_like(X, t.color[1]/255.0).astype("f4")
        B = np.full_like(X, t.color[2]/255.0).astype("f4")

        positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype("f4")
        colors = np.column_stack([R.ravel(), G.ravel(), B.ravel()]).astype("f4")

        idxs = np.arange(self.grid_size*self.grid_size).reshape(self.grid_size, self.grid_size)
        tl = idxs[:-1, :-1] # -> top left corners of each 2x2 cell which we want to divide into triangles
        tr = idxs[:-1, 1:] # -> top right corners of each 2x2 cell which we want to divide into triangles
        bl = idxs[1:, :-1] # -> bottom left corners of each 2x2 cell which we want to divide into triangles
        br = idxs[1:, 1:] # -> bottom right corners of each 2x2 cell which we want to divide into triangles

        triangles = np.stack([tl, br, tr, 
                              tl, bl, br], axis=-1) # Split a 2x2 cell into 2 triangles (6 of [n-1 , n-1] to [n-1, n-1, 6])

        indices = triangles.astype("i4").ravel()
        normals = compute_normals(indices.reshape(-1,3), positions)
        vertices = np.column_stack([positions, colors, normals]).astype("f4")
        self.mesh_vbo = self.ctx.buffer(vertices.tobytes())
        self.mesh_ibo = self.ctx.buffer(indices.tobytes())
        self.mesh_vao = self.ctx.vertex_array(self.program, 
                                              [(self.mesh_vbo, "3f 3f 3f", "in_pos", "in_color", "in_normal")], 
                                              index_buffer = self.mesh_ibo, 
                                              index_element_size = 4)   

        # Set uniform variables for the fragment shader for lighting
        light = cfg.lighting
        self.program["u_light_pos"].value     = tuple(light.light_pos)
        self.program["u_light_color"].value   = tuple(light.light_color)
        self.program["u_ambient"].value       = light.ambient
        self.program["u_spec_strength"].value = light.spec_strength
        self.program["u_shininess"].value     = light.shininess

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
