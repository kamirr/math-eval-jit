use env_logger::Env;
use math_jit::{library::Library, rpn::Program, Compiler};
use std::{hint, io::Write, time::Instant};

fn test(
    func: fn(f32, f32, f32, f32, f32, f32, &mut f32, &mut f32) -> f32,
    x: f32,
    y: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    mut sig1: f32,
    mut sig2: f32,
) {
    let res = func(x, y, a, b, c, d, &mut sig1, &mut sig2);

    println!("f(..) = {res:.4}, sig1 = {sig1:.4}, sig2 = {sig2:.4}");
}

fn bench(func: fn(f32, f32, f32, f32, f32, f32, &mut f32, &mut f32) -> f32) {
    let start = Instant::now();
    let mut sig1 = 0.5f32;
    let mut sig2 = 0.0f32;
    for _ in 0..44100 {
        hint::black_box(func(2.0, 3.0, 0.0, 0.0, 0.0, 0.0, &mut sig1, &mut sig2));
    }
    let elapsed = start.elapsed().as_secs_f32();

    println!("perf coefficient: {}", 1.0 / elapsed);
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug,cranelift_codegen=info"))
        .init();

    let mut expr = String::new();
    loop {
        expr.clear();

        print!("expression: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut expr).unwrap();

        if expr.is_empty() {
            break;
        }

        let mut program = Program::parse_from_infix(expr.as_str()).unwrap();

        log::debug!("parsed: {program:?}");
        program.propagate_constants();
        log::debug!("optimized: {program:?}");

        let library = Library::default();
        let mut compiler = Compiler::new(&library).unwrap();
        let func = compiler.compile(&program).unwrap();

        test(func, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        bench(func);
    }
}
