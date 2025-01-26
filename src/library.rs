//! Management of functions accessible to programs

use std::mem::transmute;

/// Description of a function accessible to the compiled program
///
/// Currently only functions of the form `fn(f32, [f32, ..]) -> f32` are
/// supported.
pub struct FunctionF32 {
    /// Name of the function exposed to the program
    pub name: String,
    /// Pointer to the function
    pub(crate) ptr: *const u8,
    /// Number of arguments of the function
    pub(crate) param_count: usize,
}

impl FunctionF32 {
    /// Constructs a new instance of 1-argument [`FunctionF32`]
    pub fn new_1(name: String, ptr: extern "C" fn(f32) -> f32) -> Self {
        FunctionF32 {
            name,
            ptr: ptr as *const u8,
            param_count: 1,
        }
    }

    /// Constructs a new instance of 2-argument [`FunctionF32`]
    pub fn new_2(name: String, ptr: extern "C" fn(f32, f32) -> f32) -> Self {
        FunctionF32 {
            name,
            ptr: ptr as *const u8,
            param_count: 2,
        }
    }

    /// Call this function with 1 argument
    ///
    /// Returns None if parameter count is incorrect.
    pub fn call_1(&self, x: f32) -> Option<f32> {
        if self.param_count != 1 {
            None
        } else {
            // SAFETY: param_count proves that ptr was instantiated from a
            // function of this signature.
            Some(unsafe { transmute::<_, extern "C" fn(f32) -> f32>(self.ptr)(x) })
        }
    }

    /// Call this function with 2 arguments
    ///
    /// Returns None if parameter count is incorrect.
    pub fn call_2(&self, x: f32, y: f32) -> Option<f32> {
        if self.param_count != 2 {
            None
        } else {
            // SAFETY: param_count proves that ptr was instantiated from a
            // function of this signature.
            Some(unsafe { transmute::<_, extern "C" fn(f32, f32) -> f32>(self.ptr)(x, y) })
        }
    }
}

/// Library of functions accessible to the program
pub struct Library {
    funs: Vec<FunctionF32>,
}

impl Library {
    /// Append a new function to the library
    pub fn insert(&mut self, fun: FunctionF32) {
        self.funs.push(fun);
    }

    /// Iterator over the functions
    pub fn iter(&self) -> impl Iterator<Item = &'_ FunctionF32> {
        self.funs.iter()
    }
}

/// Default implementation of the library
///
/// Provider `sin(x)`, `cos(x)` and `pow(a, b)`, as those functions are
/// accessible to all programs.
impl Default for Library {
    fn default() -> Self {
        extern "C" fn sin_(x: f32) -> f32 {
            x.sin()
        }

        extern "C" fn cos_(x: f32) -> f32 {
            x.cos()
        }

        extern "C" fn tan_(x: f32) -> f32 {
            x.tan()
        }

        extern "C" fn abs_(x: f32) -> f32 {
            x.abs()
        }

        extern "C" fn sqrt_(x: f32) -> f32 {
            x.sqrt()
        }

        extern "C" fn pow_(a: f32, b: f32) -> f32 {
            a.powf(b)
        }

        Library {
            funs: vec![
                FunctionF32::new_1("sin".into(), sin_),
                FunctionF32::new_1("cos".into(), cos_),
                FunctionF32::new_1("tan".into(), tan_),
                FunctionF32::new_1("abs".into(), abs_),
                FunctionF32::new_1("sqrt".into(), sqrt_),
                FunctionF32::new_2("pow".into(), pow_),
            ],
        }
    }
}
