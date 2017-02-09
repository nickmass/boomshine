#[macro_use]
extern crate glium;
extern crate rand;
use rand::{Rng, thread_rng as rng};

use std::ops::{Add, Mul, Sub};

fn main() {
    let mut game = Game::new();
    game.do_loop();
}

struct Game {
    bounds: Rectangle,
    targets: Vec<Target>,
    window: Window,
    mouse_down: bool,
}

impl Game {
    fn new() -> Self {
        let dims = (1024, 1024);
        let mut game = Game {
            bounds: Rectangle {
                origin: vector2(0., 0.),
                width: 100.0,
                height: 100.0,
            },
            targets: Vec::new(),
            window: Window::new(dims.0, dims.1),
            mouse_down: false,
        };

        for _ in 0..1000 {
            game.targets.push(Target::new(game.bounds));
        }
        game
    }

    fn do_loop(&mut self) {
        let mut last_tick = std::time::Instant::now();
        while !self.window.is_closed() {
            let elapased = last_tick.elapsed();
            last_tick = std::time::Instant::now();
            let dt = {
                let secs = elapased.as_secs() as f64;
                let nanos = elapased.subsec_nanos() as f64 / 1_000_000_000.0;
                secs + nanos
            };
            println!("{:?}", dt);
            self.tick(dt);
            self.render();
        }
    }

    fn tick(&mut self, dt: f64) {
        if self.targets.len() == 0 {
            for _ in 0..1000 {
                self.targets.push(Target::new(self.bounds));
            }
        }

        for mut target in self.targets.iter_mut() {
            target.tick(dt);
        }

        let living = self.targets.drain(..).filter(|t| t.state != TargetState::Dead).collect();
        self.targets = living;

        let killers: Vec<(Vector2, f64)> = self.targets
            .iter()
            .filter_map(|t| match t.state {
                TargetState::Growing(s) |
                TargetState::Shrinking(s) => Some((t.center, t.radius * s)),
                _ => None,
            })
            .collect();

        for mut target in self.targets
            .iter_mut()
            .filter(|t| t.state == TargetState::Alive)
            .filter(|t| killers.iter().any(|k| t.center.distance(k.0) < k.1 + t.radius)) {
            target.state = TargetState::Growing(1.0);
        }

        let mouse = self.window.poll_input();
        if !self.mouse_down && mouse.0 {
            let (x, y) = mouse.1;
            let center = vector2(x * self.bounds.width, y * self.bounds.height);
            let targ = Target::bomb(self.bounds, center);
            self.targets.push(targ);
        }
        self.mouse_down = mouse.0;
    }

    fn render(&mut self) {
        self.window.render::<_, _, _, Target>(&self.bounds, &self.targets);
    }
}

#[derive(Copy, Clone, Debug)]
struct Vector2 {
    x: f64,
    y: f64,
}
fn vector2(x: f64, y: f64) -> Vector2 {
    Vector2 { x: x, y: y }
}
impl Vector2 {
    fn distance(self, other: Vector2) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}
impl Mul<f64> for Vector2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}
impl Sub<f64> for Vector2 {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Vector2 {
            x: self.x - rhs,
            y: self.y - rhs,
        }
    }
}
impl Add<f64> for Vector2 {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Vector2 {
            x: self.x + rhs,
            y: self.y + rhs,
        }
    }
}
impl Add for Vector2 {
    type Output = Self;
    fn add(self, rhs: Vector2) -> Self {
        Vector2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Rectangle {
    origin: Vector2,
    width: f64,
    height: f64,
}
impl Rectangle {
    fn left(&self) -> f64 {
        self.origin.x
    }
    fn right(&self) -> f64 {
        self.origin.x + self.width
    }
    fn top(&self) -> f64 {
        self.origin.y + self.height
    }
    fn bottom(&self) -> f64 {
        self.origin.y
    }
    fn contains(&self, p: Vector2) -> bool {
        p.x >= self.left() && p.x <= self.right() && p.y <= self.top() && p.y >= self.bottom()
    }
    fn contract(&self, amount: f64) -> Rectangle {
        Rectangle {
            origin: self.origin + amount,
            width: self.width - amount * 2.0,
            height: self.height - amount * 2.0,
        }
    }
}

impl<'a> Into<Matrix3> for &'a Rectangle {
    fn into(self) -> Matrix3 {
        [[2.0 / self.width as f32, 0.0, 0.0],
         [0.0, 2.0 / self.height as f32, 0.0],
         [-1.0, -1.0, 1.0]]
    }
}

#[derive(PartialEq)]
enum TargetState {
    Alive,
    Growing(f64),
    Shrinking(f64),
    Dead,
}

struct Target {
    radius: f64,
    bounds: Rectangle,
    center: Vector2,
    state: TargetState,
    color: [f32; 4],
    direction: Vector2,
}

impl Target {
    fn new(bounds: Rectangle) -> Self {
        let sub = rng().gen_range(0x1000000, 0x3000000) << 8;
        let color = 0xffffff9f_u32 - sub;
        let r = (color >> 24 & 0xff) as f32 / 255.0;
        let g = (color >> 16 & 0xff) as f32 / 255.0;
        let b = (color >> 8 & 0xff) as f32 / 255.0;
        let a = (color >> 0 & 0xff) as f32 / 255.0;

        let radius = 2.0;
        let bounds = bounds.contract(radius);
        Target {
            radius: radius,
            bounds: bounds,
            center: vector2(rng().gen_range(bounds.left(), bounds.right()),
                            rng().gen_range(bounds.bottom(), bounds.top())),
            state: TargetState::Alive,
            color: [r, g, b, a],
            direction: vector2(rng().gen_range(-1.0, 1.0), rng().gen_range(-1.0, 1.0)),
        }
    }

    fn bomb(bounds: Rectangle, center: Vector2) -> Self {
        let color = 0xffffff9f_u32;
        let r = (color >> 24 & 0xff) as f32 / 255.0;
        let g = (color >> 16 & 0xff) as f32 / 255.0;
        let b = (color >> 8 & 0xff) as f32 / 255.0;
        let a = (color >> 0 & 0xff) as f32 / 255.0;
        Target {
            radius: 2.0,
            bounds: bounds,
            center: center,
            state: TargetState::Growing(1.0),
            color: [r, g, b, a],
            direction: vector2(0.0, 0.0),
        }
    }

    fn tick(&mut self, dt: f64) {
        match self.state {
            TargetState::Alive => {
                let mut new_point = self.center + (self.direction * dt * 5.0);
                while !self.bounds.contains(new_point) {
                    if new_point.x < self.bounds.left() {
                        new_point.x = self.bounds.left() + (self.bounds.left() - new_point.x);
                        self.direction.x *= -1.0;
                    }
                    if new_point.x > self.bounds.right() {
                        new_point.x = self.bounds.right() - (new_point.x - self.bounds.right());
                        self.direction.x *= -1.0;
                    }
                    if new_point.y < self.bounds.bottom() {
                        new_point.y = self.bounds.bottom() + (self.bounds.bottom() - new_point.y);
                        self.direction.y *= -1.0;
                    }
                    if new_point.y > self.bounds.top() {
                        new_point.y = self.bounds.top() - (new_point.y - self.bounds.top());
                        self.direction.y *= -1.0;
                    }
                }
                self.center = new_point;
            }
            TargetState::Growing(x) => {
                let scale = x + dt * 1.5;
                if scale > 10.0 {
                    self.state = TargetState::Dead;
                } else if scale > 5.0 {
                    self.state = TargetState::Shrinking(10.0 - scale);
                } else {
                    self.state = TargetState::Growing(scale);
                }
            }
            TargetState::Shrinking(x) => {
                let scale = x - dt * 8.0;
                if scale > 0.0 {
                    self.state = TargetState::Shrinking(scale);
                } else {
                    self.state = TargetState::Dead;
                }
            }
            _ => {}
        }
    }
}

impl Mesh for Target {
    fn matrix(&self) -> Matrix3 {
        let r = match self.state {
            TargetState::Growing(s) |
            TargetState::Shrinking(s) => self.radius * s,
            _ => self.radius,
        };
        [[r as f32, 0.0, 0.0],
         [0.0, r as f32, 0.0],
         [self.center.x as f32, self.center.y as f32, 1.0]]
    }

    fn color(&self) -> [f32; 4] {
        self.color.clone()
    }

    fn model_name() -> &'static str {
        "target"
    }

    fn model() -> Vec<Vertex> {
        let tri_count = 50_i32;
        let mut tris = Vec::new();
        let step = (2.0 * std::f64::consts::PI) / tri_count as f64;
        let mut accu = 0.0_f64;
        for _ in 0..tri_count {
            tris.push(Vertex { position: [0.0, 0.0] });
            tris.push(Vertex { position: [accu.sin(), accu.cos()] });
            accu += step;
            tris.push(Vertex { position: [accu.sin(), accu.cos()] });
        }
        tris
    }
}

use glium::Surface;
use glium::DisplayBuild;
use glium::vertex::VertexBuffer;
use std::collections::HashMap;
use std::borrow::Borrow;

type Matrix3 = [[f32; 3]; 3];

#[derive(Copy, Clone)]
struct Vertex {
    position: [f64; 2],
}

implement_vertex!(Vertex, position);


#[derive(Copy, Clone)]
struct TargetData {
    matrix: Matrix3,
    color: [f32; 4],
}

implement_vertex!(TargetData, matrix, color);

pub struct Window {
    display: glium::Display,
    program: glium::Program,
    closed: bool,
    mouse_down: bool,
    mouse_position: (f64, f64),
    models: HashMap<String, VertexBuffer<Vertex>>,
}

impl Window {
    pub fn new(width: u32, height: u32) -> Window {
        let dims = (width, height);
        let display = glium::glutin::WindowBuilder::new()
            .with_dimensions(dims.0, dims.1)
            .with_title(format!("Boomshine"))
            .build_glium()
            .unwrap();

        let vert_shader = r#"
            #version 140

            uniform mat3 view_matrix;
            in mat3 matrix;
            in vec4 color;
            in vec2 position;
            out vec4 v_color;

            void main() {
                vec3 p = view_matrix * matrix * vec3(position, 1.0);
                v_color = color;
                gl_Position = vec4(p.x / p.z, p.y / p.z, 1.0, 1.0);
            }
        "#;

        let frag_shader = r#"
            #version 140

            in vec4 v_color;
            out vec4 color;

            void main() {
                color = v_color;
            }
       "#;

        let program = glium::Program::from_source(&display, vert_shader, frag_shader, None)
            .unwrap();

        Window {
            display: display,
            program: program,
            closed: false,
            mouse_down: false,
            mouse_position: (0.0, 0.0),
            models: HashMap::new(),
        }
    }

    fn process_events(&mut self) {
        use glium::glutin::{Event, MouseButton, ElementState};
        for ev in self.display.poll_events() {
            match ev {
                Event::MouseInput(s, b) => {
                    self.mouse_down = b == MouseButton::Left && s == ElementState::Pressed;
                }
                Event::MouseMoved(x, y) => {
                    let dims = self.display.get_framebuffer_dimensions();
                    self.mouse_position = (x as f64 / dims.0 as f64,
                                           (dims.1 as i32 - y) as f64 / dims.1 as f64);
                }
                Event::Closed => self.closed = true,
                _ => (),
            }
        }
    }

    fn poll_input(&mut self) -> (bool, (f64, f64)) {
        self.process_events();
        (self.mouse_down, self.mouse_position)
    }

    fn render<X, I, U, M>(&mut self, view_matrix: X, items: I)
        where X: Into<Matrix3>,
              I: IntoIterator<Item = U>,
              U: Borrow<M>,
              M: Mesh
    {
        {
            let params = glium::DrawParameters {
                blend: glium::draw_parameters::Blend::alpha_blending(),
                ..Default::default()
            };
            let model = if self.models.contains_key(M::model_name().into()) {
                self.models.get(M::model_name().into()).unwrap()
            } else {
                let buf = glium::VertexBuffer::new(&self.display, &M::model()).unwrap();
                self.models.insert(M::model_name().into(), buf);
                self.models.get(M::model_name().into()).unwrap()
            };

            let mut target_data = Vec::new();

            for item in items.into_iter() {
                let item = item.borrow();
                target_data.push(TargetData {
                    matrix: item.matrix(),
                    color: item.color(),
                });
            }

            let target_data_buf = glium::VertexBuffer::new(&self.display, &target_data).unwrap();

            let view_matrix = view_matrix.into();

            let uniforms = uniform!{
                view_matrix: view_matrix,
            };

            let indicies = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

            let mut target = self.display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            target.draw((model, target_data_buf.per_instance().unwrap()),
                      &indicies,
                      &self.program,
                      &uniforms,
                      &params)
                .unwrap();
            target.finish().unwrap();
        }
        self.process_events();
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

trait Mesh {
    fn matrix(&self) -> Matrix3;
    fn color(&self) -> [f32; 4];
    fn model_name() -> &'static str;
    fn model() -> Vec<Vertex>;
}
