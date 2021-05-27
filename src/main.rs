// https://raytracing.github.io/books/RayTracingInOneWeekend.html

use rand::Rng;
use rand::RngCore;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Default)]
struct Vec3 {
    e: [f64; 3],
}

impl Vec3 {
    fn new(e0: f64, e1: f64, e2: f64) -> Vec3 {
        Vec3 { e: [e0, e1, e2] }
    }

    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    fn length_squared(&self) -> f64 {
        self[0] * self[0] + self[1] * self[1] + self[2] * self[2]
    }

    fn dot(&self, rhs: &Vec3) -> f64 {
        self[0] * rhs[0] + self[1] * rhs[1] + self[2] * rhs[2]
    }

    fn cross(&self, rhs: &Vec3) -> Vec3 {
        Vec3::new(
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        )
    }

    fn unit_vector(&self) -> Vec3 {
        *self / self.length()
    }

    fn x(&self) -> f64 {
        self[0]
    }

    fn y(&self) -> f64 {
        self[1]
    }

    fn z(&self) -> f64 {
        self[2]
    }

    fn random(rng: &mut dyn RngCore) -> Vec3 {
        Vec3::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        )
    }

    fn random_within(min: f64, max: f64, rng: &mut dyn RngCore) -> Vec3 {
        Vec3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }

    fn random_in_unit_sphere(rng: &mut dyn RngCore) -> Vec3 {
        loop {
            let p = Vec3::random_within(-1.0, 1.0, rng);
            if p.length_squared() >= 1.0 {
                continue;
            }
            return p;
        }
    }

    fn random_unit_vector(rng: &mut dyn RngCore) -> Vec3 {
        Vec3::random_in_unit_sphere(rng).unit_vector()
    }

    fn _random_in_hemisphere(normal: &Vec3, rng: &mut impl Rng) -> Vec3 {
        let in_unit_sphere = Vec3::random_in_unit_sphere(rng);
        if in_unit_sphere.dot(&normal) > 0.0 {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }

    fn near_zero(&self) -> bool {
        let s = 1e-8;
        self[0].abs() < s && self[1].abs() < s && self[2].abs() < s
    }

    fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
        (*v) - 2.0 * v.dot(n) * (*n)
    }

    fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = (-*uv).dot(n).min(1.0);
        let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
        let r_out_parallel = -(((1.0 - r_out_perp.length_squared()).abs()).sqrt()) * *n;
        r_out_perp + r_out_parallel
    }

    fn random_in_unit_disk(rng: &mut dyn RngCore) -> Vec3 {
        loop {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() >= 1.0 {
                continue;
            }
            return p;
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self[0], -self[1], -self[2])
    }
}

impl std::ops::Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, i: usize) -> &Self::Output {
        &self.e[i]
    }
}

impl std::ops::IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.e[i]
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
    }
}

impl std::ops::SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
    }
}

impl std::ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self[0] *= t;
        self[1] *= t;
        self[2] *= t;
    }
}

impl std::ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        *self *= 1.0 / t;
    }
}

impl std::fmt::Display for Vec3 {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(fmt, "{} {} {}", self[0], self[1], self[2])
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self[0] *= rhs[0];
        self[1] *= rhs[1];
        self[2] *= rhs[2];
        self
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, mut rhs: Vec3) -> Self::Output {
        rhs *= self;
        rhs
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self::Output {
        self *= 1.0 / rhs;
        self
    }
}

type Point3 = Vec3;
type Color = Vec3;

#[derive(Default)]
struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Point3, direction: Vec3) -> Ray {
        Ray { origin, direction }
    }

    fn at(&self, t: f64) -> Point3 {
        self.origin + t * self.direction
    }

    fn color(&self, world: &impl Hittable, depth: u32, rng: &mut dyn RngCore) -> Color {
        if depth == 0 {
            return Color::new(0.0, 0.0, 0.0);
        }
        if let Some(rec) = world.hit(self, 0.001, INFONITY) {
            if let Ok(scatter_result) = rec.mat_ptr.scatter(self, &rec, rng) {
                return scatter_result.attenuation
                    * scatter_result.scattered.color(world, depth - 1, rng);
            }
            return Color::new(0.0, 0.0, 0.0);
        }
        let unit_direction = self.direction.unit_vector();
        let t = 0.5 * (unit_direction.y() + 1.0);
        (1.0 - t) * Color::new(1.0, 1.0, 1.0) + t * Color::new(0.5, 0.7, 1.0)
    }
}

#[derive(Clone)]
struct HitRecord {
    p: Point3,
    normal: Vec3,
    mat_ptr: Arc<dyn Material>,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new(
        p: Point3,
        mat_ptr: Arc<dyn Material>,
        t: f64,
        r: &Ray,
        outward_normal: Vec3,
    ) -> HitRecord {
        let front_face = r.direction.dot(&outward_normal) < 0.0;
        HitRecord {
            p,
            normal: if front_face {
                outward_normal
            } else {
                -outward_normal
            },
            mat_ptr: mat_ptr.clone(),
            t,
            front_face,
        }
    }
}

trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

struct Sphere {
    center: Point3,
    radius: f64,
    material: Arc<dyn Material>,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Arc<dyn Material>) -> Sphere {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    #[allow(clippy::many_single_char_names, clippy::suspicious_operation_groupings)]
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = oc.dot(&r.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let t = root;
        let p = r.at(t);
        let outward_normal = (p - self.center) / self.radius;
        Some(HitRecord::new(
            p,
            self.material.clone(),
            t,
            r,
            outward_normal,
        ))
    }
}

#[derive(Default)]
struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    fn new() -> HittableList {
        Default::default()
    }

    fn _clear(&mut self) {
        self.objects.clear();
    }

    fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest_so_far = t_max;
        let mut result = None;
        for object in &self.objects {
            if let Some(temp_rec) = object.hit(r, t_min, closest_so_far) {
                closest_so_far = temp_rec.t;
                result = Some(temp_rec);
            }
        }
        result
    }
}

const INFONITY: f64 = f64::INFINITY;

struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    _w: Vec3,
    lens_radius: f64,
}

impl Camera {
    fn new(
        lookfrom: Point3,
        lookat: Point3,
        vup: Vec3,
        vfov: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Camera {
        let theta = vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).unit_vector();
        let u = vup.cross(&w).unit_vector();
        let v = w.cross(&u);

        let origin = lookfrom;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
        let lens_radius = aperture / 2.0;

        Camera {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
            u,
            v,
            _w: w,
            lens_radius,
        }
    }

    fn get_ray(&self, s: f64, t: f64, rng: &mut dyn RngCore) -> Ray {
        let rd = self.lens_radius * Vec3::random_in_unit_disk(rng);
        let offset = self.u * rd.x() + self.v * rd.y();
        Ray::new(
            self.origin + offset,
            self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset,
        )
    }
}

trait Material: Send + Sync {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> std::result::Result<ScatterResult, ()>;
}

struct ScatterResult {
    attenuation: Color,
    scattered: Ray,
}

struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    fn new(albedo: Color) -> Lambertian {
        Lambertian { albedo }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        _r_in: &Ray,
        rec: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> std::result::Result<ScatterResult, ()> {
        let scatter_direction = {
            let scatter_direction = rec.normal + Vec3::random_unit_vector(rng);
            if scatter_direction.near_zero() {
                rec.normal
            } else {
                scatter_direction
            }
        };
        Ok(ScatterResult {
            attenuation: self.albedo,
            scattered: Ray::new(rec.p, scatter_direction),
        })
    }
}

struct Metal {
    albedo: Color,
    fuzz: f64,
}

impl Metal {
    fn new(albedo: Color, fuzz: f64) -> Metal {
        Metal {
            albedo,
            fuzz: if fuzz < 1.0 { fuzz } else { 1.0 },
        }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> std::result::Result<ScatterResult, ()> {
        let reflected = Vec3::reflect(&r_in.direction.unit_vector(), &rec.normal);
        let scattered = Ray::new(
            rec.p,
            reflected + self.fuzz * Vec3::random_in_unit_sphere(rng),
        );
        if scattered.direction.dot(&rec.normal) > 0.0 {
            Ok(ScatterResult {
                attenuation: self.albedo,
                scattered,
            })
        } else {
            Err(())
        }
    }
}

struct Dielectric {
    ir: f64, // Index of Refaction
}

impl Dielectric {
    fn new(ir: f64) -> Dielectric {
        Dielectric { ir }
    }

    fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        // Use Schlick's approximation for reflectance.
        let r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        rng: &mut dyn RngCore,
    ) -> std::result::Result<ScatterResult, ()> {
        let refraction_ratio = if rec.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };
        let unit_direction = r_in.direction.unit_vector();

        let cos_theta = (-unit_direction).dot(&rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let direction = if cannot_refract
            || Self::reflectance(cos_theta, refraction_ratio) > rng.gen_range(0.0..1.0) as f64
        {
            Vec3::reflect(&unit_direction, &rec.normal)
        } else {
            Vec3::refract(&unit_direction, &rec.normal, refraction_ratio)
        };

        Ok(ScatterResult {
            attenuation: Color::new(1.0, 1.0, 1.0),
            scattered: Ray::new(rec.p, direction),
        })
    }
}

fn random_schene(rng: &mut dyn RngCore) -> HittableList {
    let mut world = HittableList::new();

    let ground_material = Arc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    world.add(Box::new(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    )));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen_range(0.0..1.0);
            let center = Point3::new(
                a as f64 + 0.9 * rng.gen_range(0.0..1.0),
                0.2,
                b as f64 + 0.9 * rng.gen_range(0.0..1.0),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::random(rng) * Color::random(rng);
                    let sphere_material = Arc::new(Lambertian::new(albedo));
                    world.add(Box::new(Sphere::new(center, 0.2, sphere_material)));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::random_within(0.5, 1.0, rng);
                    let fuzz = rng.gen_range(0.0..0.5);
                    let sphere_material = Arc::new(Metal::new(albedo, fuzz));
                    world.add(Box::new(Sphere::new(center, 0.2, sphere_material)));
                } else {
                    // glass
                    let sphere_material = Arc::new(Dielectric::new(1.5));
                    world.add(Box::new(Sphere::new(center, 0.2, sphere_material)));
                }
            }
        }
    }

    let material1 = Arc::new(Dielectric::new(1.5));
    world.add(Box::new(Sphere::new(
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        material1,
    )));

    let material2 = Arc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    world.add(Box::new(Sphere::new(
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        material2,
    )));

    let material3 = Arc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));
    world.add(Box::new(Sphere::new(
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        material3,
    )));

    world
}

fn render(world: &HittableList) -> (Vec<Color>, u32, u32) {
    // Image
    let aspect_ratio = 3.0 / 2.0;
    let image_width = 1200;
    let image_height = (image_width as f64 / aspect_ratio) as u32;

    let samples_per_pixel = 500;
    // let samples_per_pixel = 10;

    let max_depth = 50;

    // Camera
    let lookfrom = Point3::new(13.0, 2.0, 3.0);
    let lookat = Point3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = 10.0;
    let aperture = 0.1;

    let camera = Camera::new(
        lookfrom,
        lookat,
        vup,
        20.0,
        aspect_ratio,
        aperture,
        dist_to_focus,
    );

    // Report progress
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let total = image_width * image_height;
        let mut count = 0;
        while let Ok(_) = rx.recv() {
            count += 1;
            if count % 100 == 0 {
                eprint!("\r[{:3}%] {:8} /{:8}", count * 100 / total, count, total)
            }
        }
    });

    // Render
    let pixels = (0..image_height)
        .rev()
        .flat_map(|j| {
            (0..image_width)
                .map(|i| (j, i, tx.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<(u32, u32, _)>>()
        .into_par_iter()
        .map(|(j, i, tx)| {
            let mut rng = rand::thread_rng();
            let pixel = (0..samples_per_pixel).fold(Color::new(0.0, 0.0, 0.0), |acc, _| {
                let u = (i as f64 + rng.gen_range(0.0..1.0)) / (image_width - 1) as f64;
                let v = (j as f64 + rng.gen_range(0.0..1.0)) / (image_height - 1) as f64;
                let r = camera.get_ray(u, v, &mut rng);
                acc + r.color(world, max_depth, &mut rng)
            });
            tx.send(()).unwrap();
            pixel / samples_per_pixel as f64
        })
        .collect();
    (pixels, image_width, image_height)
}

fn main() {
    let mut rng = rand::thread_rng();

    // World
    let world = random_schene(&mut rng);
    let (pixels, image_width, image_height) = render(&world);

    println!("P3");
    println!("{} {}", image_width, image_height);
    println!("255");

    for pixel in pixels {
        let r = pixel.x().sqrt();
        let g = pixel.y().sqrt();
        let b = pixel.z().sqrt();

        println!(
            "{} {} {}",
            (256.0 * r.clamp(0.0, 0.999)) as i64,
            (256.0 * g.clamp(0.0, 0.999)) as i64,
            (256.0 * b.clamp(0.0, 0.999)) as i64,
        );
    }

    eprintln!("\nDone.");
}
