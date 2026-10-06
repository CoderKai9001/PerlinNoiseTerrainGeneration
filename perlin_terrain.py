import moderngl
import numpy as np
from pathlib import Path
import moderngl_window as mglw
import glm

rng = np.random.default_rng(seed=42)

def get_perlin_grid(dim):
    theta = rng.uniform(0.0, 2.0*np.pi, size=(dim,dim))
    grads = np.stack([np.cos(theta), np.sin(theta)], axis=-1) # (dim-1, dim-1, 2)

    return grads 

def fade_curve(x): return (6.0*np.power(x, 5) - 15*np.power(x, 4) + 10*np.power(x, 3))

def lerp(p,q,t):
    return p + t*(q-p)

def get_perlin_noise(X, Z, cells, offset=(0.0, 0.0)):
    
    # Convert the evaluation grid (terrain grid) from [-1,+1] to [0, dim-1]
    px = (X + 1.0) / 2.0 * cells + offset[0]
    pz = (Z + 1.0) / 2.0 * cells + offset[1]

    dim = int(np.ceil(cells)) + 2
    grads = get_perlin_grid(dim)

    # Limit the floor computation to inner grid as there cant be 4 neighbouring lattice points for a mesh point on the border
    cx = np.floor(px).astype(int)
    cz = np.floor(pz).astype(int)

    fx = px - cx
    fz = pz - cz

    sx = fade_curve(fx)
    sz = fade_curve(fz)

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
    noise = np.zeros_like(X, dtype=np.float64)
    total_amp = 0.0
    for k in range(num_octaves):
        amp = persistence ** k
        cells = base_cells * lacunarity ** k
        offset = rng.uniform(0.0, 1.0, size=2)   # random shift per octave
        noise += amp * get_perlin_noise(X, Z, cells, offset)
        total_amp += amp
    return noise / total_amp

class GridMesh(mglw.WindowConfig):
    gl_version = (3,3)
    title = "Perlin Terrain"
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

        self.grid_size = 200
        perlin_amplitude = 0.3
        x = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        z = np.linspace(-1.0, 1.0, self.grid_size).astype("f4")
        X, Z = np.meshgrid(x,z)
        Y = (perlin_amplitude * fbm(X,Z,5,4)).astype("f4")
        # Set all colors to white
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
