#version 330

in vec3 v_color;
in vec3 v_world_pos;
in vec3 v_normal;

uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform vec3 u_view_pos;
uniform float u_ambient;
uniform float u_spec_strength;
uniform float u_shininess;

out vec4 f_color;

void main() {
    vec3 L = normalize(u_light_pos - v_world_pos); // vector pointing towards light source
    // vec3 L = normalize(vec3(-1.0, 1.0, 1.0));
    vec3 N = normalize(v_normal); // Normal vector
    vec3 V = normalize(u_view_pos); // viewing vector i.e; vector pointing towards the camera
    vec3 H = normalize(L + V); // Half vector between L and V (used in Blinn-Phong equation for lighting)

    float diff = max(dot(N,L), 0.0);
    float spec = diff > 0.0 ? u_spec_strength * pow(max(dot(N,H), 0.0), u_shininess) : 0.0;

    vec3 ambient = u_ambient * u_light_color;
    vec3 diffuse = diff * u_light_color;
    vec3 specular = spec * u_light_color;

    vec3 color = (ambient + diffuse) * v_color + specular;
    f_color = vec4(color, 1.0);
}