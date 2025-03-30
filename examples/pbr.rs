use bevy_march::*;

use bevy::{
    core_pipeline::bloom::Bloom,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    math::vec3,
    prelude::*,
    sprite::Anchor,
    render::{render_resource::ShaderType, renderer::RenderDevice, view::RenderLayers},
    window::CursorGrabMode,
};

#[derive(Component)]
struct Offset {
    t: f32,
    scale: f32,
    speed: f32,
}

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set::<WindowPlugin>(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }),
        FrameTimeDiagnosticsPlugin,
    ));

    let main_pass_shader = app.world().resource::<AssetServer>().load("features.wgsl");

    app.add_plugins(RayMarcherPlugin::<PbrMaterial>::new(main_pass_shader))
        .init_resource::<CursorState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (grab_cursor, rotate_and_move).chain(),
                rotate_light,
                update_fps,
            ),
        )
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug)]
pub struct PbrMaterial {
    pub base_color: Vec4,       // RGB + Alpha для прозрачности
    pub emissive: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,       // F0 для диэлектриков
    pub ior: f32,               // Индекс преломления для прозрачности
    pub transmission: f32,      // Коэффициент прозрачности
    pub normal_map_strength: f32, // Для нормал-маппинга, если понадобится
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            emissive: Vec3::ZERO,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            ior: 1.5,
            transmission: 0.0,
            normal_map_strength: 1.0,
        }
    }
}

impl MarcherMaterial for PbrMaterial {
    fn is_transparent(&self) -> bool {
        self.base_color.w < 0.99 || self.transmission > 0.01
    }
}

#[derive(Component)]
struct FpsText;

#[derive(Resource, Default, PartialEq, Eq)]
enum CursorState {
    #[default]
    Free,
    Locked,
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut mats: ResMut<Assets<PbrMaterial>>,
    loader: Res<AssetServer>,
    device: Res<RenderDevice>,
) {
    // FPS текст
    commands.spawn((
        Text2d::default(),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        Anchor::TopLeft,
        FpsText,
    ));

    // Камера
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            far: 100.,
            ..default()
        }),
        Transform::from_translation(vec3(0.0, 0.0, 5.0)).looking_at(Vec3::ZERO, Vec3::Y),
        RenderLayers::from_layers(&[0, 1]),
        MarcherSettings::default(),
        MarcherMainTextures::new(&mut images, (1280, 720)),
        MarcherConeTexture::new(&mut images, &device, (1280, 720)),
        Bloom {
            intensity: 0.3,
            composite_mode: bevy::core_pipeline::bloom::BloomCompositeMode::Additive,
            prefilter: bevy::core_pipeline::bloom::BloomPrefilter {
                threshold: 1.0,
                threshold_softness: 0.0,
            },
            ..default()
        },
    ));

    // Свет
    commands.spawn((
        DirectionalLight {
            color: Color::rgb(1.0, 1.0, 0.9),
            illuminance: 10_000.,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1., 1.5, 1.).looking_at(Vec3::ZERO, Vec3::Y),
        MarcherShadowSettings::default(),
        MarcherShadowTextures::new(&mut images),
    ));

    // Загрузка SDF из примера features.rs
    let center_shape = loader.load("sdfs/features_center.sdf3d");
    let surround_shape = loader.load("sdfs/features_surround.sdf3d");
    let floor_shape = loader.load("sdfs/features_floor.sdf3d");

    // Металлические материалы
    let gold_material = mats.add(PbrMaterial {
        base_color: Vec4::new(1.0, 0.85, 0.57, 1.0),
        emissive: Vec3::ZERO,
        metallic: 1.0,
        roughness: 0.1,
        reflectance: 0.95,
        ior: 1.5,
        transmission: 0.0,
        normal_map_strength: 1.0,
    });
    
    let copper_material = mats.add(PbrMaterial {
        base_color: Vec4::new(0.95, 0.64, 0.54, 1.0),
        emissive: Vec3::ZERO,
        metallic: 1.0,
        roughness: 0.2,
        reflectance: 0.8,
        ior: 1.5,
        transmission: 0.0,
        normal_map_strength: 1.0,
    });
    
    // Стеклянный материал
    let glass_material = mats.add(PbrMaterial {
        base_color: Vec4::new(0.9, 0.9, 1.0, 0.2),
        emissive: Vec3::ZERO,
        metallic: 0.0,
        roughness: 0.05,
        reflectance: 0.5,
        ior: 1.5,
        transmission: 0.95,
        normal_map_strength: 1.0,
    });
    
    // Светящийся материал
    let emissive_material = mats.add(PbrMaterial {
        base_color: Vec4::new(1.0, 0.3, 0.1, 1.0),
        emissive: Vec3::new(2.0, 0.6, 0.2),
        metallic: 0.0,
        roughness: 0.5,
        reflectance: 0.3,
        ior: 1.5,
        transmission: 0.0,
        normal_map_strength: 1.0,
    });

    // Центральный объект (золотой)
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, -0.5, -10.0)),
        RenderedSdf {
            sdf: center_shape,
            material: gold_material,
        },
    ));

    // Окружающие объекты (медные)
    for (pos, scale, speed) in [
        (vec3(3., -1.6, -15.), 0.44, 0.8),
        (vec3(-5., -1.3, -12.), 0.35, -1.),
        (vec3(6., -1.4, -9.), 0.3, -1.2),
        (vec3(-5., -1.5, -7.), 0.5, 0.4),
    ] {
        commands.spawn((
            Transform::from_translation(pos).with_scale(Vec3::splat(0.6)),
            RenderedSdf {
                sdf: surround_shape.clone(),
                material: copper_material.clone(),
            },
            Offset {
                t: 0.,
                scale,
                speed,
            },
        ));
    }

    // Стеклянный объект
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.0, -8.0)).with_scale(Vec3::splat(0.7)),
        RenderedSdf {
            sdf: surround_shape.clone(),
            material: glass_material,
        },
    ));

    // Светящийся объект
    commands.spawn((
        Transform::from_translation(Vec3::new(-3.0, 2.0, -12.0)).with_scale(Vec3::splat(0.5)),
        RenderedSdf {
            sdf: surround_shape.clone(),
            material: emissive_material,
        },
        Offset {
            t: 0.,
            scale: 0.3,
            speed: 0.6,
        },
    ));

    // Пол с отражениями
    let floor_material = mats.add(PbrMaterial {
        base_color: Vec4::new(0.5, 0.5, 0.5, 1.0),
        emissive: Vec3::ZERO,
        metallic: 0.0,
        roughness: 0.8,
        reflectance: 0.2,
        ior: 1.5,
        transmission: 0.0,
        normal_map_strength: 1.0,
    });

    commands.spawn((
        Transform::from_xyz(0., -2.25, 0.),
        RenderedSdf {
            sdf: floor_shape.clone(),
            material: floor_material,
        },
    ));

    // Водная поверхность (прозрачная с отражениями)
    let water_material = mats.add(PbrMaterial {
        base_color: Vec4::new(0.1, 0.5, 0.8, 0.3),
        emissive: Vec3::ZERO,
        metallic: 0.0,
        roughness: 0.1,
        reflectance: 0.3,
        ior: 1.33,
        transmission: 0.9,
        normal_map_strength: 1.0,
    });

    commands.spawn((
        Transform::from_xyz(0., -2.0, 0.),
        RenderedSdf {
            sdf: floor_shape,
            material: water_material,
        },
    ));
}

fn update_fps(
    window: Single<&Window>,
    mut text: Single<(&mut Transform, &mut Text2d), With<FpsText>>,
    diag_store: Res<DiagnosticsStore>,
) {
    let half_size = window.resolution.size() * 0.5;
    let (ref mut transform, ref mut text) = *text;
    let Some(fps) = diag_store.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(fps) = fps.smoothed() else {
        return;
    };
    transform.translation = Vec3::new(-half_size.x, half_size.y, 0.);
    text.clear();
    text.push_str(&format!("FPS: {:.1}", fps))
}

fn grab_cursor(
    mut windows: Query<&mut Window>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut cursor_state: ResMut<CursorState>,
) {
    let grabbed = if keyboard_input.just_pressed(KeyCode::Escape) {
        false
    } else if mouse_input.just_pressed(MouseButton::Left) {
        true
    } else {
        return;
    };

    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };

    (window.cursor_options.grab_mode, *cursor_state) = if grabbed {
        (CursorGrabMode::Confined, CursorState::Locked)
    } else {
        (CursorGrabMode::None, CursorState::Free)
    };
    window.cursor_options.visible = !grabbed;
}

fn rotate_and_move(
    time: Res<Time>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    cursor_state: Res<CursorState>,
) {
    let rotation_input = -mouse_motion
        .read()
        .fold(Vec2::ZERO, |acc, mot| acc + mot.delta);
    let movement_input = Vec3::new(
        keyboard_input.pressed(KeyCode::KeyD) as u32 as f32
            - keyboard_input.pressed(KeyCode::KeyA) as u32 as f32,
        keyboard_input.pressed(KeyCode::Space) as u32 as f32
            - keyboard_input.pressed(KeyCode::ControlLeft) as u32 as f32,
        keyboard_input.pressed(KeyCode::KeyS) as u32 as f32
            - keyboard_input.pressed(KeyCode::KeyW) as u32 as f32,
    );

    if rotation_input.length_squared() < 0.001 && movement_input.length_squared() < 0.001 {
        return;
    }

    for mut transform in cameras.iter_mut() {
        let translation = movement_input * time.delta_secs() * 5.;
        let translation = transform.rotation * translation;
        transform.translation += translation;
        transform.translation.y = transform.translation.y.max(-1.9);

        if *cursor_state == CursorState::Locked {
            let mut euler = transform.rotation.to_euler(EulerRot::YXZ);
            euler.0 += rotation_input.x * 0.003;
            euler.1 += rotation_input.y * 0.003;
            transform.rotation = Quat::from_euler(EulerRot::YXZ, euler.0, euler.1, 0.);
        }
    }
}

fn rotate_light(time: Res<Time>, mut lights: Query<&mut Transform, With<DirectionalLight>>) {
    for mut transform in lights.iter_mut() {
        let mut euler = transform.rotation.to_euler(EulerRot::YXZ);
        euler.0 += 0.2 * time.delta_secs();
        transform.rotation = Quat::from_euler(EulerRot::YXZ, euler.0, euler.1, euler.2);
    }
}


fn update_offsets(time: Res<Time>, mut spheres: Query<(&mut Transform, &mut Offset)>) {
    for (mut transform, mut offset) in spheres.iter_mut() {
        // Remove old offset
        transform.translation.y -= offset.t.sin() * offset.speed;

        // Calculate and apply new offset
        offset.t += offset.scale * time.delta_secs();
        transform.translation.y += offset.t.sin() * offset.speed;
    }
}