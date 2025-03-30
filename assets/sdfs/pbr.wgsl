#import bevy_march::{get_individual_ray, march_ray, settings, calc_normal, get_occlusion, MarchSettings, MarchResult, depth_texture};

@group(1) @binding(2) var color_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var<storage, read> materials: array<PbrMaterial>;

const PI: f32 = 3.14159265359;

struct PbrMaterial {
    base_color: vec4<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    roughness: f32,
    reflectance: f32,
    ior: f32,
    transmission: f32,
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let march = get_individual_ray(invocation_id.xy);
    let res = march_ray(march);
    
    textureStore(depth_texture, invocation_id.xy, vec4<f32>(settings.near / res.traveled, 0., 0., 0.));
    
    if (res.traveled >= 100.) {
        textureStore(color_texture, invocation_id.xy, vec4<f32>(skybox(march.direction), 1.0));
        return;
    }
    
    let hit_point = march.origin + march.direction * res.traveled;
    let normal = calc_normal(res.id, hit_point);
    
    let material = materials[res.material];
    
    let view_dir = -march.direction;
    
    let light_dir = normalize(-settings.light_dir);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    
    let f0 = mix(
        vec3<f32>(material.reflectance), 
        material.base_color.rgb, 
        material.metallic
    );
    
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let h_dot_v = max(dot(half_dir, view_dir), 0.0);
    
    let D = distribution_ggx(n_dot_h, material.roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, material.roughness);
    let F = fresnel_schlick(h_dot_v, f0);
    
    let numerator = D * G * F;
    let denominator = max(4.0 * n_dot_v * n_dot_l, 0.001);
    let specular = numerator / denominator;
    
    let k_s = F;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - material.metallic);
    
    let diffuse = k_d * material.base_color.rgb / PI;
    
    let direct = (diffuse + specular) * light_color * n_dot_l;
    
    let ambient = vec3<f32>(0.03) * material.base_color.rgb;
    
    let emission = material.emissive;
    
    var color = direct + ambient + emission;
    
    if (material.metallic > 0.0 || material.reflectance > 0.0) {
        let reflect_dir = reflect(march.direction, normal);
        var reflect_march = march;
        reflect_march.origin = hit_point + normal * 0.01;
        reflect_march.direction = reflect_dir;
        reflect_march.start = 0.;
        reflect_march.limit = settings.far - res.traveled;
        reflect_march.ignored = res.id;
        
        let reflect_res = march_ray(reflect_march);
        let reflect_intensity = mix(material.reflectance, 1.0, material.metallic);
        
        if (reflect_res.traveled < 100.) {
            let reflect_hit = reflect_march.origin + reflect_march.direction * reflect_res.traveled;
            let reflect_normal = calc_normal(reflect_res.id, reflect_hit);
            let reflect_material = materials[reflect_res.material];
            
            let reflect_n_dot_l = max(dot(reflect_normal, light_dir), 0.0);
            let reflect_color = reflect_material.base_color.rgb * reflect_n_dot_l + reflect_material.emissive;
            
            color = mix(color, reflect_color, reflect_intensity * F);
        } else {
            color = mix(color, skybox(reflect_dir), reflect_intensity * F);
        }
    }
    
    if (material.transmission > 0.0) {
        let refract_dir = refract_ray(march.direction, normal, material.ior);
        var refract_march = march;
        refract_march.origin = hit_point + refract_dir * 0.01;
        refract_march.direction = refract_dir;
        refract_march.start = 0.;
        refract_march.limit = settings.far - res.traveled;
        refract_march.ignored = res.id;
        
        let refract_res = march_ray(refract_march);
        
        if (refract_res.traveled < 100.) {
            let refract_hit = refract_march.origin + refract_march.direction * refract_res.traveled;
            let refract_normal = calc_normal(refract_res.id, refract_hit);
            let refract_material = materials[refract_res.material];
            
            let refract_n_dot_l = max(dot(refract_normal, light_dir), 0.0);
            let refract_color = refract_material.base_color.rgb * refract_n_dot_l + refract_material.emissive;
            
            let absorption = exp(-material.base_color.rgb * (1.0 - material.transmission) * refract_res.traveled * 0.1);
            
            color = mix(color, refract_color * absorption, material.transmission);
        } else {
            let background = skybox(refract_dir);
            let absorption = exp(-material.base_color.rgb * (1.0 - material.transmission) * 5.0);
            
            color = mix(color, background * absorption, material.transmission);
        }
    }
    
    if (res.traveled > 50.) {
        let factor = min((res.traveled - 50.) / 50., 1.);
        color = mix(color, vec3<f32>(0.0), factor);
    }
    
    textureStore(color_texture, invocation_id.xy, vec4<f32>(color, material.base_color.a));
}

fn get_color(march: MarchSettings, res: MarchResult) -> vec3<f32> {
    if res.traveled >= 100. {
        return skybox(march.direction);
    }

    let hit = march.origin + march.direction * (res.traveled - 0.02);
    let normal = calc_normal(res.id, hit);
    var diffuse = dot(normal, -settings.light_dir);

    var material = materials[res.material];
    var albedo = material.base_color;
    var emission = material.emissive;
    if material.reflectance > 0.01 {
        let base_strength = (1. - material.reflectance);
        let base_color = base_strength * material.base_color;

        // TODO: Make reflections less boilerplate heavy
        var reflected = march;
        reflected.origin = march.origin + march.direction * res.traveled;
        reflected.direction = reflect(march.direction, normal);
        reflected.start = 0.;
        reflected.limit = settings.far - res.traveled;
        reflected.ignored = res.id;
        let res = march_ray(reflected);
        let refl_mat = materials[res.material];

        if res.distance < 0.1 {
            emission += refl_mat.emissive * material.reflectance;
            albedo = base_color + refl_mat.base_color * material.reflectance;

            let reflected_hit = reflected.origin + reflected.direction * (res.traveled - 0.01);
            let reflected_normal = calc_normal(res.id, reflected_hit);
            diffuse = max(dot(reflected_normal, -settings.light_dir), 0.);
        } else {
            albedo = vec4<f32>(base_color.rgb + skybox(reflected.direction) * material.reflectance, base_color.a);
        }
    }
    var ambient = 1.;
    if diffuse <= 0.15 {
        // TODO: When reflected, use final hit and normal for AO
        ambient = get_occlusion(march.origin + march.direction * res.traveled, normal);
    }
    let light = max(diffuse, ambient * 0.15);
    let color = max(emission, vec3<f32>(albedo.rgb * light));
    if res.traveled > 50. {
        let factor = min((res.traveled - 50.) / 50., 1.);
        return (1. - factor) * color;
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

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h_2 = n_dot_h * n_dot_h;
    
    let denom = n_dot_h_2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    
    let ggx1 = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx2 = n_dot_l / (n_dot_l * (1.0 - k) + k);
    
    return ggx1 * ggx2;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn refract_ray(incident: vec3<f32>, normal: vec3<f32>, ior: f32) -> vec3<f32> {
    let cos_i = dot(-incident, normal);
    let eta = select(ior, 1.0 / ior, cos_i > 0.0);
    let n = select(-normal, normal, cos_i > 0.0);
    let cos_i_abs = abs(cos_i);
    
    let k = 1.0 - eta * eta * (1.0 - cos_i_abs * cos_i_abs);
    
    if (k < 0.0) {
        return reflect(incident, n);
    }
    
    return eta * incident + (eta * cos_i_abs - sqrt(k)) * n;
}

fn pbr_lighting(
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    material: PbrMaterial
) -> vec3<f32> {
    let half_dir = normalize(view_dir + light_dir);
    
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    let h_dot_v = max(dot(half_dir, view_dir), 0.0);
    
    let f0 = mix(
        vec3<f32>(material.reflectance), 
        material.base_color.rgb, 
        material.metallic
    );
    
    let D = distribution_ggx(n_dot_h, material.roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, material.roughness);
    let F = fresnel_schlick(h_dot_v, f0);
    
    let numerator = D * G * F;
    let denominator = max(4.0 * n_dot_v * n_dot_l, 0.001);
    let specular = numerator / denominator;
    
    let k_s = F;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - material.metallic);
    
    let diffuse = k_d * material.base_color.rgb / PI;
    
    let direct = (diffuse + specular) * light_color * n_dot_l;
    
    let emission = material.emissive;
    
    let ambient = vec3<f32>(0.03) * material.base_color.rgb * (1.0 - material.metallic);
    
    return direct + ambient + emission;
}
