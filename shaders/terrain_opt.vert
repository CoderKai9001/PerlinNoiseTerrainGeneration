#version 330

in vec3 in_pos; // only x and z are used; the height is generated here

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;
uniform vec3 u_view_pos;

// fBm parameters (same meaning as the terrain section of the config)
uniform uint  u_seed;
uniform int   u_num_octaves;
uniform float u_base_cells;
uniform float u_persistence;
uniform float u_lacunarity;
uniform float u_amplitude;
uniform int   u_cavity_octaves; // octaves that make up the smooth base the cavity term is measured against

out vec3 v_world_pos;
out vec3 v_normal;
out float v_cavity;

const float TAU = 6.28318530718;

// PCG hash
uint pcg_hash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Chris Wellons' lowbias32
uint lowbias32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// uint -> float in [0, 1): put 23 random bits into the mantissa of 1.0
float u2f(uint h) {
    return uintBitsToFloat((h >> 9) | 0x3f800000u) - 1.0;
}

// Random unit gradient at lattice point `cell`; replaces the CPU's precomputed gradient grid.
// Chaining the hash over (octave key, x, z) gives every lattice point of every octave its own angle.
vec2 gradient(ivec2 cell, uint octave_key) {
    uint h = pcg_hash(uint(cell.y) + pcg_hash(uint(cell.x) + octave_key));
    float theta = u2f(h) * TAU;
    return vec2(cos(theta), sin(theta));
}

// 2D Perlin noise at lattice-space point p. Returns (noise, d noise/dp.x, d noise/dp.y).
vec3 perlin_noise(vec2 p, uint octave_key) {
    ivec2 c = ivec2(floor(p));
    vec2 f = p - vec2(c);

    vec2 g00 = gradient(c,               octave_key);
    vec2 g10 = gradient(c + ivec2(1, 0), octave_key);
    vec2 g01 = gradient(c + ivec2(0, 1), octave_key);
    vec2 g11 = gradient(c + ivec2(1, 1), octave_key);

    float n00 = dot(g00, f);
    float n10 = dot(g10, f - vec2(1.0, 0.0));
    float n01 = dot(g01, f - vec2(0.0, 1.0));
    float n11 = dot(g11, f - vec2(1.0, 1.0));

    vec2 u  = f * f * f * (f * (6.0 * f - 15.0) + 10.0); // 6f^5 - 15f^4 + 10f^3
    vec2 du = 30.0 * f * f * (f - 1.0) * (f - 1.0);      // its derivative

    // Bilinear blend written out so it can be differentiated:
    // n = n00 + u.x*(n10-n00) + u.y*(n01-n00) + u.x*u.y*k
    float k = n00 - n10 - n01 + n11;
    float n = n00 + u.x * (n10 - n00) + u.y * (n01 - n00) + u.x * u.y * k;

    // Each corner term n_ij = dot(g_ij, f - corner) has derivative g_ij; the blend weights add the du terms
    vec2 dn = g00 + u.x * (g10 - g00) + u.y * (g01 - g00) + u.x * u.y * (g00 - g10 - g01 + g11)
            + du * vec2(n10 - n00 + u.y * k, n01 - n00 + u.x * k);

    return vec3(n, dn);
}

// Fractal Brownian motion over the terrain's [-1, 1] xz range. Returns (height, d height/dx, d height/dz).
// `detail` is what the octaves after the first u_cavity_octaves add on top of that smooth base:
// negative in creases and gullies (the point is lower than its surroundings), positive on ridges.
vec3 fbm(vec2 xz, out float detail) {
    vec3 sum = vec3(0.0);
    float base = 0.0;
    float amp = 1.0;
    float cells = u_base_cells;
    float total_amp = 0.0;
    for (int k = 0; k < u_num_octaves; k++) {
        uint octave_key = pcg_hash(u_seed + uint(k));
        vec2 offset = vec2(u2f(pcg_hash(octave_key)), u2f(pcg_hash(octave_key + 1u))); // random shift per octave

        // [-1, 1] -> [0, cells] lattice space; dp/dxz = cells / 2 rescales the derivative back to world units
        vec2 p = (xz + 1.0) * 0.5 * cells + offset;
        vec3 n = perlin_noise(p, octave_key);
        sum += amp * vec3(n.x, n.yz * cells * 0.5);
        if (k == u_cavity_octaves - 1) base = sum.x;

        total_amp += amp;
        amp *= u_persistence;
        cells *= u_lacunarity;
    }
    detail = (sum.x - base) / total_amp;
    return sum / total_amp;
}

void main() {
    float detail;
    vec3 h = u_amplitude * fbm(in_pos.xz, detail);
    vec3 pos = vec3(in_pos.x, h.x, in_pos.z);

    // Surface y = h(x, z) has normal (-dh/dx, 1, -dh/dz)
    vec3 normal = normalize(vec3(-h.y, 1.0, -h.z));

    v_world_pos = vec3(u_model * vec4(pos, 1.0));
    v_normal = u_normal_mat * normal;
    v_cavity = detail; // left unscaled by u_amplitude so the darkening strength doesn't depend on terrain height
    gl_Position = u_mvp * vec4(pos, 1.0);
}
