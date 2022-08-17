


VERTEX_SHADER_BASE = b"""
#version 330
layout(location = 0) in vec3 a_pos;
layout(location = 2) in vec2 a_texCoord;

uniform mat4 a_projection;
uniform mat4 a_model;

out vec2 TexCoord;

void main()
{
    gl_Position = a_projection * a_model * vec4(a_pos, 1.0f);
    TexCoord = a_texCoord;
}
"""

FRAGMENT_SHADER_BASE = b"""
#version 330
in vec3 ourColor;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
    vec4 color1 = texture(texture1, TexCoord);
    vec4 color2 = texture(texture2, TexCoord);
    //vec4 hi = vec(40.588, 0.0, 1.0, 1.0);
    vec4 hi = vec4(0.5, 0.95, 1.0, 1.0);
    FragColor = mix(color1, hi, color2.r);
    //FragColor = mix(color1, color2, 0.5);
}
"""

FRAGMENT_SHADER_COMPUTE = b"""
#version 330
in vec2 TexCoord;

//out vec4 FragColor;
layout(location = 0) out vec4 FragColor;

#define ROOTTWO 1.414

uniform vec2 resolution;

uniform sampler2D video_src;
uniform sampler2D prev_frame;

uniform float time;
uniform float seed;
uniform float decay;
uniform float reaction;
uniform int highlight;
uniform float threshold;

uniform vec3 clusterCenters[18];
uniform vec3 center;


float rand(vec2 n) {
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}


float when_gt(float x, float y) {
    return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
    return max(sign(y - x), 0.0);
}


float distSquared( vec3 A, vec3 B )
{
    vec3 C = A - B;
    return dot( C, C );
}


float getSource(vec2 uv){
    float g = texture(video_src, uv).g;
	float x = when_gt(g, threshold);

    return x; //vec4(x, 0, 0, x);
}


float getSource2(vec2 uv){
    // returns the index of the closest color
    vec3 c = texture(video_src, uv).rgb;

    float minDist = 100000.0;
    int argMin = 0;
    int i;
    for (i = 0; i < 6; i++){
        float distance = distSquared(c, clusterCenters[i]);
        if(distance < minDist){
            minDist = distance;
            argMin = i;
        }
        //minDist = min(distance, minDist);
    }

    if (argMin == highlight){
        return 1.0;
    } else {
        return 0.0;
    }
}

float getSource3(vec2 uv){
    vec3 c = texture(video_src, uv).rgb;
    float distance = distSquared(c, center);

    float x = when_lt(distance, threshold);
    return x;
}

void main(){
    vec2 res = 1.0 / resolution;
    vec2 uv = gl_FragCoord.xy * res;


    vec4 color = texture(prev_frame, uv);

    float r = 0.0;
	r += texture(prev_frame, uv + vec2(-ROOTTWO,-ROOTTWO)*res).r;
	r += texture(prev_frame, uv + vec2( 0,-1)*res).r;
	r += texture(prev_frame, uv + vec2( ROOTTWO,-ROOTTWO)*res).r;
	r += texture(prev_frame, uv + vec2( 1, 0)*res).r;
	r += texture(prev_frame, uv + vec2( ROOTTWO, ROOTTWO)*res).r;
	r += texture(prev_frame, uv + vec2( 0, 1)*res).r;
	r += texture(prev_frame, uv + vec2(-ROOTTWO, ROOTTWO)*res).r;
	r += texture(prev_frame, uv + vec2(-1, 0)*res).r;

	float tr = rand(uv + vec2(time*3, time*0.2));

    //float cr = color.r;
    color.r *= decay;

    //color.g = highlight/255.0;

    color.r += getSource3(uv);

    // color.r += source*seed

    float T = 0.5;

    if (r > 0.5 + 0.7*T && tr > 0.5 + 0.45 * -cos(uv.y*3.0+ time*0.5) && color.r < reaction/200){
        color.r = 1.0;
    }

    FragColor = color;
}
"""

VERTEX_SHADER_COMPUTE = b"""
#version 330
layout(location = 0) in vec3 a_pos;
layout(location = 2) in vec2 a_texCoord;


out vec2 TexCoord;

void main()
{
    gl_Position = vec4(a_pos, 1.0f);
    TexCoord = a_texCoord;
}
"""




# ================== ES ==============




VERTEX_SHADER_BASE_ES = b"""
#version 300 es

in vec3 a_pos;
in vec3 a_color;
in vec2 a_texCoord;

uniform mat4 a_projection;
uniform mat4 a_model;

out vec2 TexCoord;
out vec3 OurColor;

void main()
{
    gl_Position = a_projection * a_model * vec4(a_pos, 1.0f);
    TexCoord = a_texCoord;
    OurColor = a_color;
}
"""

FRAGMENT_SHADER_BASE_ES = b"""
#version 300 es
precision mediump float;

in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform mediump float shade;

void main()
{
    vec4 color1 = texture(texture1, TexCoord);
    vec4 color2 = texture(texture2, TexCoord);
    vec4 hi = vec4(vec3(shade), 1.0);
    FragColor = mix(color1, hi, color2.r);
    //FragColor = mix(color1, color2, 0.5);
}
"""

FRAGMENT_SHADER_COMPUTE_ES = b"""
#version 300 es
precision mediump float;

in vec2 TexCoord;

out vec4 FragColor;
//layout(location = 0) out vec4 FragColor;

uniform vec2 resolution;

uniform sampler2D video_src;
uniform sampler2D prev_frame;

uniform float time;
uniform float seed;
uniform float decay;
uniform float reaction;
uniform float threshold;


float rand(vec2 n) {
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}


float when_gt(float x, float y) {
    return max(sign(x - y), 0.0);
}

float getSource(vec2 uv){
    float g = texture(video_src, uv).g;
	float x = when_gt(g, threshold);

    return x; //vec4(x, 0, 0, x);
}

void main()
{
    vec2 res = 1.0 / resolution;
    vec2 uv = gl_FragCoord.xy * res;
    float source = getSource(uv);

    vec4 color = texture(prev_frame, uv);

    float r = 0.0;
	r += texture(prev_frame, uv + vec2(-1.0,-1.0)*res).r;
	r += texture(prev_frame, uv + vec2( 0.0,-1.0)*res).r;
	r += texture(prev_frame, uv + vec2( 1.0,-1.0)*res).r;
	r += texture(prev_frame, uv + vec2( 1.0, 0.0)*res).r;
	r += texture(prev_frame, uv + vec2( 1.0, 1.0)*res).r;
	r += texture(prev_frame, uv + vec2( 0.0, 1.0)*res).r;
	r += texture(prev_frame, uv + vec2(-1.0, 1.0)*res).r;
	r += texture(prev_frame, uv + vec2(-1.0, 0.0)*res).r;

	float tr = rand(uv + vec2(time*3.0, time*0.2));

    //float cr = color.r;
    color.r *= decay;
    
    color.r += source * seed;

    float T = 1.0;

    if (r > 0.5 + 0.7*T && tr > 0.5 + 0.45 * -cos(uv.y*3.0+ time*0.5) && color.r < reaction){
        color.r = 1.0;
    }

    //if (color.r > 0.0) {
    //    color.g = 1.0;
    //} else {
    //    color.g = 0.0;
    //}


    FragColor = color;
    }
"""

VERTEX_SHADER_COMPUTE_ES = b"""
#version 300 es

precision mediump float;

in vec3 a_pos;
in vec3 a_color;
in vec2 a_texCoord;


out vec2 TexCoord;

void main()
{
    gl_Position = vec4(a_pos, 1.0f);
    TexCoord = a_texCoord;
}
"""

