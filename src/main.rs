use std::{hint, time::Instant};

use math_eval_jit::{library::Library, rpn::Program, Compiler};

fn test(
    func: fn(f32, f32, f32, f32, f32, f32, &mut f32, &mut f32) -> f32,
    in1: f32,
    in2: f32,
    alpha: f32,
    beta: f32,
    delta: f32,
    gamma: f32,
    mut sig1: f32,
    mut sig2: f32,
) {
    let res = func(in1, in2, alpha, beta, delta, gamma, &mut sig1, &mut sig2);

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
    let expr = "sin(sin(in1) + in2 * (pi/4))";

    println!("expression: {expr}");

    let mut program = Program::parse_from_infix(expr).unwrap();

    println!("parsed: {program:?}");
    program.propagate_constants();
    println!("optimized: {program:?}");

    let library = Library::default();
    let mut compiler = Compiler::new(&library).unwrap();
    let func = compiler.compile(&program).unwrap();

    test(func, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    bench(func);
}
