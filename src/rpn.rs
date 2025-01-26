//! Parsing and operations on the program

use crate::error::JitError;

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
                    "sig1" => Token::PushVar(Var::Sig1),
                    "sig2" => Token::PushVar(Var::Sig2),
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
    pub fn propagate_constants(&mut self) {
        let mut work_done = true;
        while work_done {
            work_done = false;

            if self.0.len() < 2 {
                continue;
            }

            for n in 0..self.0.len() - 1 {
                let tok0 = &self.0[n];
                let tok1 = &self.0[n + 1];

                let arg = match tok0 {
                    Token::Push(f) => f.value(),
                    _ => continue,
                };

                let result = match tok1 {
                    Token::Function(Function { name, args: 1 }) if name == "sin" => arg.sin(),
                    Token::Function(Function { name, args: 1 }) if name == "cos" => arg.cos(),
                    Token::Unop(Unop::Neg) => -arg,
                    _ => continue,
                };

                self.0[n] = Token::Noop;
                self.0[n + 1] = Token::Push(Value::Literal(result));
                work_done = true;
            }

            if self.0.len() < 3 {
                continue;
            }

            for n in 0..self.0.len() - 2 {
                let tok0 = &self.0[n];
                let tok1 = &self.0[n + 1];
                let tok2 = &self.0[n + 2];

                let (l, r) = match (tok0, tok1) {
                    (Token::Push(l), Token::Push(r)) => (l.value(), r.value()),
                    _ => continue,
                };

                let result = match tok2 {
                    Token::Binop(Binop::Add) => l + r,
                    Token::Binop(Binop::Sub) => l - r,
                    Token::Binop(Binop::Mul) => l * r,
                    Token::Binop(Binop::Div) => l / r,
                    Token::Function(Function { name, args: 2 }) if name == "pow" => l.powf(r),
                    _ => continue,
                };

                self.0[n] = Token::Noop;
                self.0[n + 1] = Token::Noop;
                self.0[n + 2] = Token::Push(Value::Literal(result));
                work_done = true;
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
        Program,
    };

    use super::{Binop, Function, Unop, Var};

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
                "sin(cos(tan(2)))",
                vec![
                    two(),
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
}
