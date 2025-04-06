use bevy_march::*;

use bevy::{
    core_pipeline::bloom::Bloom,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    math::vec3,
    prelude::*,
    render::{render_resource::ShaderType, renderer::RenderDevice, view::RenderLayers},
    sprite::Anchor,
    text::TextBounds,
    window::CursorGrabMode,
};

fn make_single_threaded(schedule: &mut Schedule) {
    schedule.set_executor_kind(bevy::ecs::schedule::ExecutorKind::SingleThreaded);
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
    ))
    .edit_schedule(PreUpdate, make_single_threaded)
    .edit_schedule(Update, make_single_threaded)
    .edit_schedule(PostUpdate, make_single_threaded);

    let main_pass_shader = app.world().resource::<AssetServer>().load("tectonic_plates.wgsl");

    app.add_plugins(RayMarcherPlugin::<SdfMaterial>::new(main_pass_shader))
        .init_resource::<CursorState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (grab_cursor, rotate_and_move).chain(),
                update_offsets,
                rotate_light,
                update_fps,
            ),
        )
        .run();
}

#[derive(Asset, ShaderType, TypePath, Clone, Debug, Default)]
struct SdfMaterial {
    base_color: Vec3,
    metallic: f32,
    roughness: f32,
    emissive: Vec3,
    subsurface_color: Vec3,
    subsurface_thickness: f32,
    transparency: f32,         
    ior: f32,                   
    absorption: Vec3,           
}

impl MarcherMaterial for SdfMaterial {}

#[derive(Component)]
struct FpsText;
#[derive(Component)]
struct HelpText;

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut mats: ResMut<Assets<SdfMaterial>>,
    loader: Res<AssetServer>,
    device: Res<RenderDevice>,
) {
    commands.spawn((
        Text2d::default(),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        Anchor::TopLeft,
        FpsText,
    ));

    let mut text = Text2d::default();
    text.push_str(
        "Use WASD to move.\n\
        Space and Ctrl to go up and down.\n\
        Left click and Escape to lock and release the cursor",
    );
    commands.spawn((
        text,
        TextFont {
            font_size: 60.0,
            ..default()
        },
        TextLayout {
            justify: JustifyText::Center,
            ..default()
        },
        Anchor::Center,
        TextBounds {
            width: Some(1200.),
            height: None,
        },
        HelpText,
    ));

    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            hdr: true,
            clear_color: ClearColorConfig::None,
            ..default()
        },
    ));

    // Camera
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
        Transform::from_translation(vec3(0.0, 1.0, 0.0)).looking_at(Vec3::X, Vec3::Y),
        RenderLayers::from_layers(&[0, 1]),
        MarcherSettings::default(),
        MarcherMainTextures::new(&mut images, (8, 8)),
        MarcherConeTexture::new(&mut images, &device, (8, 8)),
        MarcherScale(1),
        Bloom {
            intensity: 0.5,
            composite_mode: bevy::core_pipeline::bloom::BloomCompositeMode::Additive,
            prefilter: bevy::core_pipeline::bloom::BloomPrefilter {
                threshold: 1.,
                threshold_softness: 0.0,
            },
            ..default()
        },
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1., 1., 0.9),
            illuminance: 5_000.,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1., 1.5, 1.).looking_at(Vec3::ZERO, Vec3::Y),
        MarcherShadowSettings::default(),
        MarcherShadowTextures::new(&mut images),
    ));

    let world_shape = loader.load("sdfs/cuboid.sdf3d");
    let sss_material = mats.add(SdfMaterial {
        base_color: LinearRgba::rgb(1.0, 1.0, 1.0).to_vec3(), 
        metallic: 0.0, 
        roughness: 0.8,
        emissive: LinearRgba::BLACK.to_vec3(),
        subsurface_color:LinearRgba::BLACK.to_vec3(),
        subsurface_thickness: 0.0,
        transparency: 0.0,
        ior: 0.0,
        absorption: LinearRgba::BLACK.to_vec3(),
    });

    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(1.0)),
        RenderedSdf {
            sdf: world_shape,
            material: sss_material,
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

#[derive(Resource, Default, PartialEq, Eq)]
enum CursorState {
    #[default]
    Free,
    Locked,
}

fn grab_cursor(
    mut windows: Query<&mut Window>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_input: Res<ButtonInput<MouseButton>>,
    mut cursor_state: ResMut<CursorState>,
    mut text: Query<&mut Visibility, With<HelpText>>,
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
    let Ok(mut help_vis) = text.get_single_mut() else {
        return;
    };

    (window.cursor_options.grab_mode, *cursor_state, *help_vis) = if grabbed {
        (
            CursorGrabMode::Confined,
            CursorState::Locked,
            Visibility::Hidden,
        )
    } else {
        (
            CursorGrabMode::None,
            CursorState::Free,
            Visibility::Inherited,
        )
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
        transform.translation.y = transform.translation.y.max(-2.);

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

#[derive(Component)]
struct Offset {
    t: f32,
    scale: f32,
    speed: f32,
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
