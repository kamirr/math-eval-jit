# Math-JIT
[![Crates.io][crates-badge]][crates-url]
[![MIT licensed][mit-badge]][mit-url]
![CI][ci-url]

[crates-badge]: https://img.shields.io/crates/v/math-jit.svg
[crates-url]: https://crates.io/crates/math-jit
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg
[mit-url]: https://github.com/kamirr/math-jit/blob/main/LICENSE
[ci-url]: https://github.com/kamirr/math-jit/actions/workflows/rust.yml/badge.svg

Math-JIT is a limited-scope implementation of a JIT compiler using
[cranelift](https://cranelift.dev/). It compiles arithmetic expressions for the
host architecture and exposes a function pointer to the result.

## Functionality
The expression parsing is implemented in [meval](https://docs.rs/meval/latest/meval).
The common arithmetic operations supported by the compiler are:
- Binary:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Powers
- Unary:
  - Negation
- Functions:
  - Sine, cosine, tangent
  - Absolute value
  - Square root

The expressions can utilize 8 variables, values of which are supplied by the
caller: `x`, `y`, `a`, `b`, `c`, `d`, `_1` and `_2`. `_1` and `_2` are
in-out variables -- they can be overriden by calling special functions `_1(..)`
and `_2(..)`, which set the values of the two signals to that of their arguments.
All reads of the signals observe the value before any writes, and the ordering
of writes is unspecified.

## Extendability
New 1+ argument functions can be added. Operators and syntax cannot be extended,
and 0-argument functions don't work due to a bug in the parser.

## What for
Walking the AST or evaluating RPN manually is slow. I'm working on a real-time
[modular software synthesiser](https://en.wikipedia.org/wiki/Modular_synthesizer),
[Modal](https://github.com/kamirr/modal), which requires 44100 executions per
second for multiple modules. Interpreters are sufficient in such cases, but
greatly lower the margin of safety.

Other than that, I found the toy language example in cranelift to be quite
unapproachable for a complete beginner like myself. This project may prove more
useful as introductory material for the codegen-related part of compilers.

## Optimizations
After translating expressions to RPN, simple constant folding is performed and
variable liveliness is minimized. Functions from the library will be called
during constant folding under the assumption that they're pure.

## Code please
```rust
use math_jit::{Program, Compiler, Library};

let program = Program::parse_from_infix("x * 2 + _1(y)").unwrap();
let mut compiler = Compiler::new(&Library::default()).unwrap();

let func = compiler.compile(&program).unwrap();
let mut sig1 = 0.0;
let mut sig2 = 0.0;
let result = func(3.0, 1.0, 0.0, 0.0, 0.0, 0.0, &mut sig1, &mut sig2);

assert_eq!(result, 7.0);
assert_eq!(sig1, 1.0);
assert_eq!(sig2, 0.0);
```