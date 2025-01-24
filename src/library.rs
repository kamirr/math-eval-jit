pub struct FunctionF32 {
    pub name: String,
    pub(crate) ptr: *const u8,
    pub(crate) param_count: usize,
}

impl FunctionF32 {
    pub fn new_1(name: String, ptr: extern "C" fn(f32) -> f32) -> Self {
        FunctionF32 {
            name,
            ptr: ptr as *const u8,
            param_count: 1,
        }
    }

    pub fn new_2(name: String, ptr: extern "C" fn(f32, f32) -> f32) -> Self {
        FunctionF32 {
            name,
            ptr: ptr as *const u8,
            param_count: 2,
        }
    }
}

pub struct Library {
    funs: Vec<FunctionF32>,
}

impl Library {
    pub fn insert(&mut self, fun: FunctionF32) {
        self.funs.push(fun);
    }

    pub fn iter(&self) -> impl Iterator<Item = &'_ FunctionF32> {
        self.funs.iter()
    }
}

impl Default for Library {
    fn default() -> Self {
        extern "C" fn sin_(x: f32) -> f32 {
            x.sin()
        }

        extern "C" fn cos_(x: f32) -> f32 {
            x.cos()
        }

        extern "C" fn pow_(a: f32, b: f32) -> f32 {
            a.powf(b)
        }

        Library {
            funs: vec![
                FunctionF32::new_1("sin".into(), sin_),
                FunctionF32::new_1("cos".into(), cos_),
                FunctionF32::new_2("pow".into(), pow_),
            ],
        }
    }
}
