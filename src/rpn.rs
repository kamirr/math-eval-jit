use crate::error::JitError;

/// RPN Operation
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Token {
    /// Push a value onto the stack
    Push(f32),
    /// Push variable value onto the stack
    PushVar(Var),
    /// Binary operation
    ///
    /// Pops 2 values from the stack, performs the operation, and pushes the
    /// result back onto the stack
    Binop(Binop),
    /// Unary operation
    ///
    /// Replaces the top value on the stack with the result of the operation
    Unop(Unop),
    /// Write top of stack to in-out variable
    Write(Out),
    /// No operation
    Noop,
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
    /// Power
    Pow,
}

/// Unary operation
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Unop {
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Negation
    Neg,
}

/// Readable variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Var {
    // Inputs
    In1,
    In2,
    Alpha,
    Beta,
    Delta,
    Gamma,
    Sig1,
    Sig2,
    // Constants
    Pi,
    E,
}

impl Var {
    /// Returns the value associated with the variable if it's constant
    pub fn as_constant(self) -> Option<f32> {
        match self {
            Var::Pi => Some(std::f32::consts::PI),
            Var::E => Some(std::f32::consts::E),
            _ => None,
        }
    }
}

/// Writeable variables
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Out {
    Sig1,
    Sig2,
}

#[derive(Debug)]
pub struct Program(pub Vec<Token>);

impl Program {
    pub fn new(tokens: Vec<Token>) -> Self {
        Program(tokens)
    }

    pub fn parse_from_infix(expr: &str) -> Result<Self, JitError> {
        let tokens = meval::tokenizer::tokenize(expr)?;
        let meval_rpn = meval::shunting_yard::to_rpn(&tokens)?;

        let mut prog = Vec::new();
        for token in meval_rpn {
            use meval::tokenizer::Operation as MevalOp;
            use meval::tokenizer::Token as MevalToken;
            let op = match token {
                MevalToken::Var(name) => {
                    let name = name.to_ascii_lowercase();
                    Token::PushVar(match name.as_str() {
                        "in1" => Var::In1,
                        "in2" => Var::In2,
                        "alpha" => Var::Alpha,
                        "beta" => Var::Beta,
                        "delta" => Var::Delta,
                        "gamma" => Var::Gamma,
                        "sig1" => Var::Sig1,
                        "sig2" => Var::Sig2,
                        "pi" => Var::Pi,
                        "e" => Var::E,
                        _ => return Err(JitError::ParseUnknownVariable(name.to_string())),
                    })
                }
                MevalToken::Number(f) => Token::Push(f as f32),
                MevalToken::Binary(op) => match op {
                    MevalOp::Plus => Token::Binop(Binop::Add),
                    MevalOp::Minus => Token::Binop(Binop::Sub),
                    MevalOp::Times => Token::Binop(Binop::Mul),
                    MevalOp::Div => Token::Binop(Binop::Div),
                    MevalOp::Pow => Token::Binop(Binop::Pow),
                    _ => return Err(JitError::ParseUnknownBinop(format!("{op:?}"))),
                },
                MevalToken::Unary(op) => match op {
                    MevalOp::Plus => Token::Noop,
                    MevalOp::Minus => Token::Unop(Unop::Neg),
                    _ => return Err(JitError::ParseUnknownUnop(format!("{op:?}"))),
                },
                MevalToken::Func(name, Some(1)) => match name.to_lowercase().as_str() {
                    "_1" => Token::Write(Out::Sig1),
                    "_2" => Token::Write(Out::Sig2),
                    "sin" => Token::Unop(Unop::Sin),
                    "cos" => Token::Unop(Unop::Cos),
                    _ => return Err(JitError::ParseUnknownFunc(name.to_string())),
                },
                other => return Err(JitError::ParseUnknownToken(format!("{other:?}"))),
            };

            prog.push(op);
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
                let tok0 = self.0[n];
                let tok1 = self.0[n + 1];

                let arg = match tok0 {
                    Token::Push(f) => f,
                    Token::PushVar(var_f) => {
                        if let Some(f) = var_f.as_constant() {
                            f
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                let result = match tok1 {
                    Token::Unop(Unop::Sin) => arg.sin(),
                    Token::Unop(Unop::Cos) => arg.cos(),
                    Token::Unop(Unop::Neg) => -arg,
                    _ => continue,
                };

                self.0[n] = Token::Noop;
                self.0[n + 1] = Token::Push(result);
                work_done = true;
            }

            if self.0.len() < 3 {
                continue;
            }

            for n in 0..self.0.len() - 2 {
                let tok0 = self.0[n];
                let tok1 = self.0[n + 1];
                let tok2 = self.0[n + 2];

                let (l, r) = match (tok0, tok1) {
                    (Token::Push(l), Token::Push(r)) => (l, r),
                    (Token::Push(l), Token::PushVar(var_r)) => {
                        if let Some(r) = var_r.as_constant() {
                            (l, r)
                        } else {
                            continue;
                        }
                    }
                    (Token::PushVar(var_l), Token::Push(r)) => {
                        if let Some(l) = var_l.as_constant() {
                            (l, r)
                        } else {
                            continue;
                        }
                    }
                    (Token::PushVar(var_l), Token::PushVar(var_r)) => {
                        if let (Some(l), Some(r)) = (var_l.as_constant(), var_r.as_constant()) {
                            (l, r)
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };

                let result = match tok2 {
                    Token::Binop(Binop::Add) => l + r,
                    Token::Binop(Binop::Sub) => l - r,
                    Token::Binop(Binop::Mul) => l * r,
                    Token::Binop(Binop::Div) => l / r,
                    Token::Binop(Binop::Pow) => l.powf(r),
                    _ => continue,
                };

                self.0[n] = Token::Noop;
                self.0[n + 1] = Token::Noop;
                self.0[n + 2] = Token::Push(result);
                work_done = true;
            }

            self.0.retain(|tok| *tok != Token::Noop);
        }
    }
}
