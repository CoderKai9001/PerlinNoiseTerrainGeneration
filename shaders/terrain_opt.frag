#version 330

in vec3 v_world_pos;
in vec3 v_normal;
in float v_cavity;

uniform vec3 u_terrain_color;

uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform vec3 u_view_pos;
uniform float u_ambient;
uniform float u_spec_strength;
uniform float u_shininess;
uniform float u_cavity_strength;

out vec4 f_color;

void main() {
    vec3 L = normalize(u_light_pos - v_world_pos); // Point Light model.
    // vec3 L = normalize(u_light_pos); // Sky light model where all light has the same vector.
    // vec3 L = normalize(vec3(-1.0, 1.0, 1.0));
    vec3 N = normalize(v_normal); // Normal vector
    vec3 V = normalize(u_view_pos - v_world_pos); // viewing vector i.e; vector pointing towards the camera
    vec3 H = normalize(L + V); // Half vector between L and V (used in Blinn-Phong equation for lighting)

    float diff = max(dot(N,L), 0.0);
    float spec = diff > 0.0 ? u_spec_strength * pow(max(dot(N,H), 0.0), u_shininess) : 0.0;

    // blend by normal's Y => lighter if normal is up or surface is flatter ; darker if steeper -> Cheap fake shadow effect 
    vec3 sky    = vec3(0.55, 0.65, 0.85);
    vec3 ground = vec3(0.25, 0.20, 0.15);
    vec3 ambient = u_ambient * mix(ground, sky, N.y * 0.5 + 0.5);

    vec3 diffuse = diff * u_light_color;
    vec3 specular = spec * u_light_color;

    // Fake ambient occlusion: points lying below their smoothed surroundings (creases, gullies) see less sky.
    // Only darkens (occ <= 1); ridges keep full light. Ambient is darkened fully, direct light partially.
    float occ = clamp(1.0 + v_cavity * u_cavity_strength, 0.0, 1.0);

    vec3 color = (ambient * occ + diffuse * mix(0.5, 1.0, occ)) * u_terrain_color + specular * occ;
    f_color = vec4(color, 1.0);
}