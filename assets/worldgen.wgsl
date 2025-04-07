#import bevy_march::{get_individual_ray, march_ray, settings, calc_normal, get_occlusion, MarchSettings, MarchResult, depth_texture}

@group(1) @binding(2) var color_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var<storage, read> materials: array<Material>;

struct Material {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
    subsurface_color: vec3<f32>,
    subsurface_thickness: f32,
    transparency: f32,
    ior: f32,
    absorption: vec3<f32>,
}

// Constants
const PI: f32 = 3.141592653589793;
const EPSILON: f32 = 0.0001;
const MAX_PLATES: u32 = 8; // Number of tectonic plates to generate

// Plate data structure
struct TectonicPlate {
    center: vec3<f32>,
    color: vec3<f32>,
    influence: f32,
}

// Global plate data
var<private> plates: array<TectonicPlate, MAX_PLATES>;

// Hash function for random number generation
fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.z) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Initialize tectonic plates with random points inside the SDF cube
fn init_plates(world_sdf: f32, world_size: f32) {
    // Seed for random generation
    let seed = 42.0;
    
    for (var i: u32 = 0u; i < MAX_PLATES; i++) {
        // Generate random points that are inside the SDF (distance <= 0)
        var point: vec3<f32>;
        var distance: f32 = 1.0;
        
        // Keep trying until we find a point inside the SDF
        for (var attempt: u32 = 0u; attempt < 100u; attempt++) {
            // Generate random point within world bounds
            let rx = hash(vec3<f32>(f32(i), f32(attempt), seed)) * 2.0 - 1.0;
            let ry = hash(vec3<f32>(f32(i), seed, f32(attempt))) * 2.0 - 1.0;
            let rz = hash(vec3<f32>(seed, f32(i), f32(attempt))) * 2.0 - 1.0;
            
            point = vec3<f32>(rx, ry, rz) * world_size;
            
            // Check if point is inside the SDF (this is a simplified check for a cube)
            distance = max(abs(point.x), max(abs(point.y), abs(point.z))) - world_size;
            
            if (distance <= 0.0) {
                break;
            }
        }
        
        // Generate a unique color for this plate
        let hue = f32(i) / f32(MAX_PLATES);
        let plate_color = hsv_to_rgb(vec3<f32>(hue, 0.8, 0.9));
        
        // Set plate data
        plates[i] = TectonicPlate(
            point,
            plate_color,
            0.5 + hash(vec3<f32>(f32(i), seed, seed)) * 0.5 // Random influence factor
        );
    }
}

// Convert HSV to RGB for plate coloring
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;
    
    let c = v * s;
    let x = c * (1.0 - abs(fract(h * 6.0) - 3.0 - 1.0));
    let m = v - c;
    
    var rgb: vec3<f32>;
    
    if (h < 1.0/6.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.0/6.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.0/6.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.0/6.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.0/6.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m);
}

// Determine which plate a point belongs to using Voronoi
fn get_plate_id(pos: vec3<f32>) -> u32 {
    var min_dist = 1000.0;
    var plate_id: u32 = 0u;
    
    for (var i: u32 = 0u; i < MAX_PLATES; i++) {
        let dist = distance(pos, plates[i].center) / plates[i].influence;
        if (dist < min_dist) {
            min_dist = dist;
            plate_id = i;
        }
    }
    
    return plate_id;
}

// Modified SDF function that divides the world into tectonic plates
fn world_sdf_with_plates(pos: vec3<f32>, world_size: f32) -> vec3<f32> {
    // Base world SDF (cube)
    let base_sdf = max(abs(pos.x), max(abs(pos.y), abs(pos.z))) - world_size;
    
    // Get the plate this position belongs to
    let plate_id = get_plate_id(pos);
    
    // Return both the SDF value and the plate color
    return vec3<f32>(base_sdf, f32(plate_id), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    // Initialize plates
    init_plates(0.0, 1.0); // Assuming a cube of size 1.0
    
    let march = get_individual_ray(invocation_id.xy);
    let res = march_ray(march);
    
    // Get hit position
    let hit = march.origin + march.direction * (res.traveled - EPSILON);
    
    // Get normal at hit point
    let N = calc_normal(res.id, hit);
    
    // Determine which plate the hit point belongs to
    var color: vec3<f32>;
    
    if (res.traveled >= settings.far) {
        // Sky color if no hit
        color = vec3<f32>(0.4, 0.6, 1.0);
    } else {
        // Get plate ID and use its color
        let plate_id = get_plate_id(hit);
        color = plates[plate_id].color;
        
        // Add some lighting
        let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
        let diffuse = max(dot(N, light_dir), 0.2);
        color = color * diffuse;
        
        // Add plate boundary visualization
        // Calculate distance to nearest plate boundary for edge highlighting
        var second_min_dist = 1000.0;
        var min_dist = 1000.0;
        
        for (var i: u32 = 0u; i < MAX_PLATES; i++) {
            let dist = distance(hit, plates[i].center) / plates[i].influence;
            if (dist < min_dist) {
                second_min_dist = min_dist;
                min_dist = dist;
            } else if (dist < second_min_dist) {
                second_min_dist = dist;
            }
        }
        
        // Highlight boundaries
        let boundary_factor = smoothstep(0.0, 0.1, abs(second_min_dist - min_dist));
        color = mix(vec3<f32>(0.1, 0.1, 0.1), color, boundary_factor);
    }
    
    textureStore(depth_texture, invocation_id.xy, vec4<f32>(settings.near / res.traveled, 0., 0., 0.));
    textureStore(color_texture, invocation_id.xy, vec4<f32>(color, 1.));
}