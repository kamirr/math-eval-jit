//! Parsing and operations on the program

use crate::{error::JitError, Library};

/// RPN Token
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Token {
    /// Push a value onto the stack
    Push(Value),
    /// Push variable value onto the stack
    PushVar(Var),
    /// Write top of stack to in-out variable
    Write(Out),
    /// Binary operation
    ///
    /// Pops 2 values from the stack, performs the operation, and pushes the
    /// result back onto the stack
    Binop(Binop),
    /// Unary operation
    ///
    /// Replaces the top value on the stack with the result of the operation
    Unop(Unop),
    /// Function call
    ///
    /// Pops a number of arguments from the stack, evaluates the function, and
    /// pushes the result back onto the stack.
    Function(Function),
    /// No operation
    Noop,
}

/// Constant value
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Value {
    /// Arbotrary value
    Literal(f32),
    /// Pi
    Pi,
    /// Euler's constant
    E,
}

impl Value {
    /// Obtains the corresponding value
    pub fn value(self) -> f32 {
        match self {
            Value::Literal(f) => f,
            Value::Pi => std::f32::consts::PI,
            Value::E => std::f32::consts::E,
        }
    }
}

/// Readable variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Var {
    X,
    Y,
    A,
    B,
    C,
    D,
    Sig1,
    Sig2,
}

/// Writeable variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Out {
    Sig1,
    Sig2,
}

/// Binary operation
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Binop {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
}

/// Unary operation
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Unop {
    /// Negation
    Neg,
}

/// Function call
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Function {
    /// Name of the function
    pub name: String,
    /// Number of arguments
    pub args: usize,
}

/// Parsed program representation
///
/// The program is represented using Reverse Polish Notation, which is lends
/// to easy iterative translation into CLIF as well as to simple optimizations.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct Program(pub Vec<Token>);

impl Program {
    /// Constructs program directly from RPN
    pub fn new(tokens: Vec<Token>) -> Self {
        Program(tokens)
    }

    /// Parses an infix notation into RPN
    pub fn parse_from_infix(expr: &str) -> Result<Self, JitError> {
        let tokens = meval::tokenizer::tokenize(expr)?;
        let meval_rpn = meval::shunting_yard::to_rpn(&tokens)?;

        let mut prog = Vec::new();
        for meval_token in meval_rpn {
            use meval::tokenizer::Operation as MevalOp;
            use meval::tokenizer::Token as MevalToken;
            let token = match meval_token {
                MevalToken::Var(name) => match name.as_str() {
                    "x" => Token::PushVar(Var::X),
                    "y" => Token::PushVar(Var::Y),
                    "a" => Token::PushVar(Var::A),
                    "b" => Token::PushVar(Var::B),
                    "c" => Token::PushVar(Var::C),
                    "d" => Token::PushVar(Var::D),
                    "_1" => Token::PushVar(Var::Sig1),
                    "_2" => Token::PushVar(Var::Sig2),
                    "pi" => Token::Push(Value::Pi),
                    "e" => Token::Push(Value::E),
                    _ => return Err(JitError::ParseUnknownVariable(name.to_string())),
                },
                MevalToken::Number(f) => Token::Push(Value::Literal(f as f32)),
                MevalToken::Binary(op) => match op {
                    MevalOp::Plus => Token::Binop(Binop::Add),
                    MevalOp::Minus => Token::Binop(Binop::Sub),
                    MevalOp::Times => Token::Binop(Binop::Mul),
                    MevalOp::Div => Token::Binop(Binop::Div),
                    MevalOp::Pow => Token::Function(Function {
                        name: "pow".to_string(),
                        args: 2,
                    }),
                    _ => return Err(JitError::ParseUnknownBinop(format!("{op:?}"))),
                },
                MevalToken::Unary(op) => match op {
                    MevalOp::Plus => Token::Noop,
                    MevalOp::Minus => Token::Unop(Unop::Neg),
                    _ => return Err(JitError::ParseUnknownUnop(format!("{op:?}"))),
                },
                MevalToken::Func(name, Some(1)) if name == "_1" => Token::Write(Out::Sig1),
                MevalToken::Func(name, Some(1)) if name == "_2" => Token::Write(Out::Sig2),
                MevalToken::Func(name, args) => Token::Function(Function {
                    name,
                    args: args.unwrap_or_default(),
                }),

                other => return Err(JitError::ParseUnknownToken(format!("{other:?}"))),
            };

            prog.push(token);
        }

        Ok(Program(prog))
    }

    /// Evaluate constant expressions
    ///
    /// Optimizes binary and unary operations:
    /// - replace `[const0, const1, op]` with `[op(const0, const1)]`
    /// - replace `[const, op]` with `[op(const)]`
    ///
    /// This operation is repeated until no progress can be made. [`Token::Noop`]
    /// is removed in the process.
    ///
    /// Doesn't support reordering of associative operations, so
    /// `[var, const0, add, const1, add]` is *not* replaced with
    /// `[var, add(const0, const1), add]` and so on.
    pub fn propagate_constants(&mut self, library: &Library) {
        let mut work_done = true;
        while work_done {
            work_done = false;

            for n in 2..self.0.len() {
                match self.0[n].clone() {
                    Token::Unop(unop) => {
                        let Token::Push(a) = self.0[n - 1] else {
                            continue;
                        };
                        let result = match unop {
                            Unop::Neg => -a.value(),
                        };

                        self.0[n - 1] = Token::Noop;
                        self.0[n] = Token::Push(Value::Literal(result));
                        work_done = true;
                    }
                    Token::Binop(binop) => {
                        let Token::Push(a) = self.0[n - 2] else {
                            continue;
                        };
                        let Token::Push(b) = self.0[n - 1] else {
                            continue;
                        };

                        let (a, b) = (a.value(), b.value());
                        let result = match binop {
                            Binop::Add => a + b,
                            Binop::Sub => a - b,
                            Binop::Mul => a * b,
                            Binop::Div => a / b,
                        };

                        self.0[n - 2] = Token::Noop;
                        self.0[n - 1] = Token::Noop;
                        self.0[n] = Token::Push(Value::Literal(result));
                        work_done = true;
                    }
                    Token::Function(Function { name, args }) => {
                        let Some(extern_fun) = library.iter().find(|f| f.name == name) else {
                            log::warn!("No function {name} in library, compilation will fail");
                            continue;
                        };

                        let result = match args {
                            1 => {
                                let Token::Push(a) = self.0[n - 1] else {
                                    continue;
                                };
                                extern_fun.call_1(a.value())
                            }
                            2 => {
                                let Token::Push(a) = self.0[n - 2] else {
                                    continue;
                                };
                                let Token::Push(b) = self.0[n - 1] else {
                                    continue;
                                };
                                extern_fun.call_2(a.value(), b.value())
                            }
                            _ => continue,
                        };

                        let Some(value) = result else {
                            log::warn!("Function {name} called with invalid number of arguments, compilation will fail");
                            continue;
                        };

                        self.0[n - args..n].fill_with(|| Token::Noop);
                        self.0[n] = Token::Push(Value::Literal(value));
                    }
                    _ => continue,
                }
            }

            self.0.retain(|tok| *tok != Token::Noop);
        }

        self.0.retain(|tok| *tok != Token::Noop);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        rpn::{Token, Value},
        Library, Program,
    };

    use super::{Binop, Function, Out, Unop, Var};

    #[test]
    fn test_parse() {
        let two = || Token::Push(Value::Literal(2.0));

        let cases = [
            ("2", vec![two()]),
            ("2 + 2", vec![two(), two(), Token::Binop(Binop::Add)]),
            ("2 - 2", vec![two(), two(), Token::Binop(Binop::Sub)]),
            ("2 * 2", vec![two(), two(), Token::Binop(Binop::Mul)]),
            ("2 / 2", vec![two(), two(), Token::Binop(Binop::Div)]),
            (
                "2 ^ 2",
                vec![
                    two(),
                    two(),
                    Token::Function(Function {
                        name: "pow".into(),
                        args: 2,
                    }),
                ],
            ),
            ("-2", vec![two(), Token::Unop(Unop::Neg)]),
            (
                "sin(cos(tan(_2(_1(2)))))",
                vec![
                    two(),
                    Token::Write(Out::Sig1),
                    Token::Write(Out::Sig2),
                    Token::Function(Function {
                        name: "tan".into(),
                        args: 1,
                    }),
                    Token::Function(Function {
                        name: "cos".into(),
                        args: 1,
                    }),
                    Token::Function(Function {
                        name: "sin".into(),
                        args: 1,
                    }),
                ],
            ),
            ("x", vec![Token::PushVar(Var::X)]),
            ("y", vec![Token::PushVar(Var::Y)]),
            ("a", vec![Token::PushVar(Var::A)]),
            ("b", vec![Token::PushVar(Var::B)]),
            ("c", vec![Token::PushVar(Var::C)]),
            ("d", vec![Token::PushVar(Var::D)]),
            ("pi", vec![Token::Push(Value::Pi)]),
            ("e", vec![Token::Push(Value::E)]),
        ];

        for (expr, tokens) in cases {
            assert_eq!(Program::parse_from_infix(expr).unwrap(), Program(tokens));
        }
    }

    #[test]
    fn test_optimize() {
        let x = |x| Token::Push(Value::Literal(x));

        fn rough_compare(prog0: &Program, prog1: &Program) -> bool {
            if prog0.0.len() != prog1.0.len() {
                return false;
            }

            for (tok0, tok1) in prog0.0.iter().zip(prog1.0.iter()) {
                const EPS: f32 = 0.00001;
                match (tok0, tok1) {
                    (Token::Push(Value::Literal(l)), Token::Push(Value::Literal(r))) => {
                        if (l - r).abs() > EPS {
                            return false;
                        }
                    }
                    (left, right) => {
                        if left != right {
                            return false;
                        }
                    }
                }
            }

            true
        }

        let cases = [
            ("2", vec![x(2.0)]),
            ("2 + 2", vec![x(4.0)]),
            ("2 + -2", vec![x(0.0)]),
            ("sin(pi/2 + pi/2)", vec![x(0.0)]),
            (
                "sin(pi/2 + pi/2) + x",
                vec![x(0.0), Token::PushVar(Var::X), Token::Binop(Binop::Add)],
            ),
        ];

        for (expr, tokens) in cases {
            let mut program = Program::parse_from_infix(expr).unwrap();
            program.propagate_constants(&Library::default());
            let expected = Program(tokens);
            assert!(
                rough_compare(&program, &expected),
                "{program:?} != {expected:?}"
            );
        }
    }
}
