#version 330

in vec3 coord;
in vec3 in_color;

uniform mat4 u_mvp;

out vec3 v_color;

void main() {
    gl_Position = u_mvp * vec4(coord.x, coord.y, coord.z, 1.0);
    v_color = in_color;
}
