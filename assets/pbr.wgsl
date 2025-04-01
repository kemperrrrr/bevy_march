#import bevy_march::{get_individual_ray, march_ray, settings, calc_normal, get_occlusion, MarchSettings, MarchResult, depth_texture};

@group(1) @binding(2) var color_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var<storage, read> materials: array<Material>;

struct Material {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
}

const PI: f32 = 3.141592653589793;
const RECIPROCAL_PI: f32 = 0.3183098861837907;
const EPSILON: f32 = 0.0001;

// PBR functions
fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let num = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let num = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = geometrySchlickGGX(NdotV, roughness);
    let ggx2 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let march = get_individual_ray(invocation_id.xy);
    let res = march_ray(march);
    let color = get_color(march, res);

    textureStore(depth_texture, invocation_id.xy, vec4<f32>(settings.near / res.traveled, 0., 0., 0.));
    textureStore(color_texture, invocation_id.xy, vec4<f32>(color, 1.));
}

fn get_color(march: MarchSettings, res: MarchResult) -> vec3<f32> {
    if res.traveled >= settings.far {
        return skybox(march.direction);
    }

    let hit = march.origin + march.direction * (res.traveled - EPSILON);
    let N = calc_normal(res.id, hit);
    let V = -normalize(march.direction);
    
    let material = materials[res.material];
    var albedo = material.base_color;
    let metallic = material.metallic;
    let roughness = material.roughness;
    let emission = material.emissive;

    // PBR lighting calculations
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    var Lo = vec3<f32>(0.0);
    
    // Directional light calculation
    let L = -settings.light_dir;
    let H = normalize(V + L);
    
    let distance = 1.0;
    let attenuation = 1.0 / (distance * distance);
    let radiance = vec3<f32>(1.0) * attenuation;
    
    // Cook-Torrance BRDF
    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - metallic;
    
    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + EPSILON;
    let specular = numerator / denominator;
    
    let NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo * RECIPROCAL_PI + specular) * radiance * NdotL;
    
    // Reflection calculation
    var reflection_color = vec3<f32>(0.0);
    if metallic > 0.01 || roughness < 0.9 {
        var reflected = march;
        reflected.origin = hit;
        reflected.direction = reflect(march.direction, N);
        reflected.start = 0.0;
        reflected.limit = settings.far - res.traveled;
        reflected.ignored = res.id;
        let reflection_res = march_ray(reflected);
        
        if reflection_res.distance < 0.1 {
            let reflection_mat = materials[reflection_res.material];
            // Включаем эмиссию в расчет цвета отражения
            reflection_color = calculate_pbr_color(
                reflected,
                reflection_res,
                reflection_mat.base_color,
                reflection_mat.metallic,
                reflection_mat.roughness,
                reflection_mat.emissive
            ) + reflection_mat.emissive; // Добавляем эмиссию дополнительно
        } else {
            reflection_color = skybox(reflected.direction);
        }
    }

    // Ambient lighting with occlusion
    let ambient_occlusion = get_occlusion(hit, N);
    let ambient = vec3<f32>(0.03) * albedo * ambient_occlusion;
    
    // Combine everything - включаем emission в основной цвет
    var color = ambient + Lo + emission;
    
    // Add reflections (factor depends on Fresnel and roughness)
    let reflection_factor = mix(0.04, 1.0, metallic) * (1.0 - roughness);
    color += reflection_color * reflection_factor;
    
    // Distance fog
    if res.traveled > 50.0 {
        let factor = min((res.traveled - 50.0) / 50.0, 1.0);
        color = mix(color, skybox(march.direction), factor);
    }
    
    return color;
}

fn calculate_pbr_color(
    march: MarchSettings,
    res: MarchResult,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>
) -> vec3<f32> {
    if res.traveled >= settings.far {
        return skybox(march.direction);
    }

    let hit = march.origin + march.direction * (res.traveled - EPSILON);
    let N = calc_normal(res.id, hit);
    let V = -normalize(march.direction);
    
    var albedo = base_color;
    
    // PBR lighting calculations
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    var Lo = vec3<f32>(0.0);
    
    // Directional light calculation
    let L = -settings.light_dir;
    let H = normalize(V + L);
    
    let distance = 1.0;
    let attenuation = 1.0 / (distance * distance);
    let radiance = vec3<f32>(1.0) * attenuation;
    
    // Cook-Torrance BRDF
    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - metallic;
    
    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + EPSILON;
    let specular = numerator / denominator;
    
    let NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo * RECIPROCAL_PI + specular) * radiance * NdotL;
    
    // Ambient lighting with occlusion
    let ambient_occlusion = get_occlusion(hit, N);
    let ambient = vec3<f32>(0.03) * albedo * ambient_occlusion;
    
    var color = ambient + Lo + emissive;

    // Tonemapping
    color = aces_tonemap_optimized(color * 1.5);
    
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));
    
    // Distance fog
    if res.traveled > 50.0 {
        let factor = min((res.traveled - 50.0) / 50.0, 1.0);
        color = mix(color, skybox(march.direction), factor);
    }
    
    return color;
}

fn skybox(direction: vec3<f32>) -> vec3<f32> {
    let sphere_uv = healpix(direction);
    let cell = floor(sphere_uv * 5.);
    let uv = (sphere_uv - cell * 0.2) * 5. - 0.5;

    var dist = 999.;
    for (var i = 0; i < 4; i++) {
        let relative = vec2<f32>(f32(i) % 2. * 2. - 1., 2. * floor(f32(i) * 0.5) - 1.);
        let pos = uv - relative * 0.5;
        let origin = hash2(cell * 2. + relative) - 0.5;
        let corner_dist = sd_star(pos - origin, 0.03, 4u, 3.);
        dist = min(dist, corner_dist);
    }

    let star = -sign(dist) * 2.;

    let noise = perlinNoise2(sphere_uv * 2.);

    let background = vec3<f32>(
        0.,
        0.,
        noise * 0.005,
    );

    return max(background, vec3<f32>(star));
}

// From https://www.shadertoy.com/view/4sjXW1
fn healpix(p: vec3<f32>) -> vec2<f32> {
    let a = atan(p.x / p.z) * 0.63662;
    let h = 3.*abs(p.y);
    var h2 = .75*p.y;
    var uv = vec2<f32>(a + h2, a - h2);
    h2 = sqrt(3. - h);
    let a2 = h2 * fract(a);
    uv = mix(uv, vec2(-h2 + a2, a2), step(2., h));

    return uv;
}

fn sd_star(pos: vec2<f32>, r: f32, n: u32, m: f32) -> f32 {
    var p = pos;
    // next 4 lines can be precomputed for a given shape
    let  an = 3.141593/f32(n);
    let  en = 3.141593/m;  // m is between 2 and n
    let acs = vec2<f32>(cos(an), sin(an));
    let ecs = vec2<f32>(cos(en), sin(en)); // ecs=vec2(0,1) for regular polygon

    let bn = modulo(atan(p.x / p.y), (2.0*an)) - an;
    p = length(p) * vec2<f32>(cos(bn), abs(sin(bn)));
    p -= r * acs;
    p += ecs * clamp(-dot(p, ecs), 0.0, r*acs.y/ecs.y);
    return length(p)*sign(p.x);
}

fn modulo(x: f32, y: f32) -> f32 {
    return x - y * floor(x/y);
}

fn hash2(in: vec2<f32>) -> vec2<f32> {
    // procedural white noise
    return fract(sin(vec2<f32>(
        dot(in, vec2(127.1,311.7)),
        dot(in, vec2(269.5,183.3))
    )) * 43758.5453);
}

// MIT License. © Stefan Gustavson, Munrocket
//
fn permute4(x: vec4<f32>) -> vec4<f32> { return ((x * 34. + 1.) * x) % vec4<f32>(289.); }
fn fade2(t: vec2<f32>) -> vec2<f32> { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2<f32>) -> f32 {
  var Pi: vec4<f32> = floor(P.xyxy) + vec4<f32>(0., 0., 1., 1.);
  let Pf = fract(P.xyxy) - vec4<f32>(0., 0., 1., 1.);
  Pi = Pi % vec4<f32>(289.); // To avoid truncation effects in permutation
  let ix = Pi.xzxz;
  let iy = Pi.yyww;
  let fx = Pf.xzxz;
  let fy = Pf.yyww;
  let i = permute4(permute4(ix) + iy);
  var gx: vec4<f32> = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
  let gy = abs(gx) - 0.5;
  let tx = floor(gx + 0.5);
  gx = gx - tx;
  var g00: vec2<f32> = vec2<f32>(gx.x, gy.x);
  var g10: vec2<f32> = vec2<f32>(gx.y, gy.y);
  var g01: vec2<f32> = vec2<f32>(gx.z, gy.z);
  var g11: vec2<f32> = vec2<f32>(gx.w, gy.w);
  let norm = 1.79284291400159 - 0.85373472095314 *
      vec4<f32>(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
  g00 = g00 * norm.x;
  g01 = g01 * norm.y;
  g10 = g10 * norm.z;
  g11 = g11 * norm.w;
  let n00 = dot(g00, vec2<f32>(fx.x, fy.x));
  let n10 = dot(g10, vec2<f32>(fx.y, fy.y));
  let n01 = dot(g01, vec2<f32>(fx.z, fy.z));
  let n11 = dot(g11, vec2<f32>(fx.w, fy.w));
  let fade_xy = fade2(Pf.xy);
  let n_x = mix(vec2<f32>(n00, n01), vec2<f32>(n10, n11), vec2<f32>(fade_xy.x));
  let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}

fn aces_tonemap_optimized(color: vec3<f32>) -> vec3<f32> {
    let m1 = mat3x3f(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    );
    
    let m2 = mat3x3f(
        1.60475, -0.53108, -0.07367,
        -0.10208, 1.10813, -0.00605,
        -0.00327, -0.07276, 1.07602
    );
    
    let v = m1 * color;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(m2 * (a / b), vec3(0.0), vec3(1.0));
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let numerator = color * (a * color + b);
    let denominator = color * (c * color + d) + e;
    return clamp(numerator / denominator, vec3(0.0), vec3(1.0));
}
