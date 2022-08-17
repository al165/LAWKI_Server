#version 130

in vec2 uv_texcoord;

uniform vec3      iResolution;
uniform sampler2D iChannel0;
uniform vec4      params;

out vec4 uv_density;

#define tx(p)  texture(iChannel0, p).xy

void main()
{
    vec2 pixelSize = 1.0 / iResolution.xy;
    vec2 cen = tx(uv_texcoord);
    vec2 rig = tx(uv_texcoord + vec2( pixelSize.x,          0.0));
    vec2 top = tx(uv_texcoord + vec2(         0.0,  pixelSize.y));
    vec2 lef = tx(uv_texcoord + vec2(-pixelSize.x,          0.0));
    vec2 bot = tx(uv_texcoord + vec2(         0.0, -pixelSize.y));

    float Du   = params.x;
    float Dv   = params.y;
    float feed = params.z;// * tx2(uv_texcoord).x;
    float kill = params.w;

    vec2 lapl = rig + top + lef + bot - 4.0 * cen;
    float du = Du * lapl.x - cen.x * cen.y * cen.y + feed * (1.0 - cen.x);
    float dv = Dv * lapl.y + cen.x * cen.y * cen.y - (feed + kill) * cen.y;
    vec2 newValue = cen + 0.6 * vec2(du, dv);

    uv_density = vec4(newValue, 0.0, 1.0);
}
