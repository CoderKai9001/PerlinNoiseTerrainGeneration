#version 330

in vec3 in_pos;
in vec3 in_color;
in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;
uniform vec3 u_view_pos;

out vec3 v_color;
out vec3 v_world_pos;
out vec3 v_normal;

void main() {
    v_world_pos = vec3(u_model * vec4(in_pos, 1.0));
    v_normal = u_normal_mat * in_normal;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_color = in_color;
}
