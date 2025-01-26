#![doc = include_str!("../README.md")]

pub mod error;
pub mod library;
pub mod rpn;

use std::collections::HashMap;

use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module};
use cranelift::prelude::{
    types::F32, AbiParam, Configurable, FunctionBuilder, FunctionBuilderContext, InstBuilder,
    MemFlags, Signature,
};
use cranelift_codegen::{ir, settings, Context};

pub use error::JitError;
pub use library::Library;
pub use rpn::Program;

/// RPN JIT compiler
pub struct Compiler {
    module: JITModule,
    module_ctx: Context,
    builder_ctx: FunctionBuilderContext,
    fun_sigs: Vec<(String, Signature)>,
}

impl Compiler {
    /// New instance of the compiler
    ///
    /// The entries in the library are made available to the programs compiled
    /// later on.
    pub fn new(library: &Library) -> Result<Self, JitError> {
        let flags = [
            ("use_colocated_libcalls", "false"),
            ("is_pic", "false"),
            ("opt_level", "speed"),
            ("enable_alias_analysis", "true"),
        ];

        let mut flag_builder = settings::builder();
        for (flag, value) in flags {
            flag_builder.set(flag, value)?;
        }

        let isa_builder =
            cranelift_native::builder().map_err(JitError::CraneliftHostUnsupported)?;

        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());
        for fun in library.iter() {
            builder.symbol(&fun.name, fun.ptr);
        }

        let module = JITModule::new(builder);
        let module_ctx = module.make_context();
        let builder_ctx = FunctionBuilderContext::new();

        let mut fun_sigs = Vec::new();
        for fun in library.iter() {
            let mut sig = module.make_signature();
            for _ in 0..fun.param_count {
                sig.params.push(AbiParam::new(F32));
            }
            sig.returns.push(AbiParam::new(F32));
            fun_sigs.push((fun.name.clone(), sig));
        }

        Ok(Compiler {
            module,
            module_ctx,
            builder_ctx,
            fun_sigs,
        })
    }

    /// Compile a [`Program`] returning a function pointer
    pub fn compile(
        &mut self,
        program: &Program,
    ) -> Result<fn(f32, f32, f32, f32, f32, f32, &mut f32, &mut f32) -> f32, JitError> {
        let ptr_type = self.module.target_config().pointer_type();

        self.module_ctx.func.signature.params = vec![
            AbiParam::new(F32),
            AbiParam::new(F32),
            AbiParam::new(F32),
            AbiParam::new(F32),
            AbiParam::new(F32),
            AbiParam::new(F32),
            AbiParam::new(ptr_type),
            AbiParam::new(ptr_type),
        ];
        self.module_ctx.func.signature.returns = vec![AbiParam::new(F32)];

        let id = self.module.declare_function(
            "jit_main",
            Linkage::Export,
            &self.module_ctx.func.signature,
        )?;

        let mut builder = FunctionBuilder::new(&mut self.module_ctx.func, &mut self.builder_ctx);

        let block = builder.create_block();
        builder.seal_block(block);

        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);

        let (v_x, v_y, v_a, v_b, v_c, v_d, v_sig1, v_sig2) = {
            let params = builder.block_params(block);
            (
                params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                params[7],
            )
        };

        let mut v_sig1_rd = None;
        let mut v_sig2_rd = None;

        let extern_funs = {
            let mut tmp = HashMap::new();
            for (name, sig) in &self.fun_sigs {
                let callee = self.module.declare_function(&name, Linkage::Import, &sig)?;
                let fun_ref = self.module.declare_func_in_func(callee, builder.func);

                tmp.insert(name.as_str(), (fun_ref, sig.params.len()));
            }

            tmp
        };

        let mut stack = Vec::new();

        for token in &program.0 {
            use rpn::{Binop, Function, Out, Token, Unop, Var};

            match token {
                Token::Push(v) => {
                    let val = builder.ins().f32const(v.value());

                    stack.push(val);
                }
                Token::PushVar(var) => {
                    let val = match var {
                        // ins
                        Var::X => v_x,
                        Var::Y => v_y,
                        Var::A => v_a,
                        Var::B => v_b,
                        Var::C => v_c,
                        Var::D => v_d,
                        // inouts
                        Var::Sig1 => *v_sig1_rd.get_or_insert_with(|| {
                            builder.ins().load(F32, MemFlags::new(), v_sig1, 0)
                        }),
                        Var::Sig2 => *v_sig2_rd.get_or_insert_with(|| {
                            builder.ins().load(F32, MemFlags::new(), v_sig2, 0)
                        }),
                    };

                    stack.push(val);
                }
                Token::Binop(op) => {
                    let b = stack
                        .pop()
                        .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;
                    let a = stack
                        .pop()
                        .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;

                    let val = match op {
                        Binop::Add => builder.ins().fadd(a, b),
                        Binop::Sub => builder.ins().fsub(a, b),
                        Binop::Mul => builder.ins().fmul(a, b),
                        Binop::Div => builder.ins().fdiv(a, b),
                    };

                    stack.push(val);
                }
                Token::Unop(op) => {
                    let x = stack
                        .pop()
                        .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;
                    let val = match op {
                        Unop::Neg => builder.ins().fneg(x),
                    };

                    stack.push(val);
                }
                Token::Write(out) => {
                    let x = *stack
                        .last()
                        .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;
                    let ptr = match out {
                        Out::Sig1 => v_sig1,
                        Out::Sig2 => v_sig2,
                    };
                    builder.ins().store(MemFlags::new(), x, ptr, 0);
                }
                Token::Function(Function { name, args }) => {
                    let (func, param_n) = *extern_funs
                        .get(name.as_str())
                        .ok_or_else(|| JitError::CompileUknownFunc(name.clone()))?;

                    // Ensure that invalid RPN won't result in an invalid function call
                    if param_n != *args {
                        return Err(JitError::CompileFuncArgsMismatch(
                            name.to_string(),
                            param_n,
                            *args,
                        ));
                    }

                    let mut arg_vs = Vec::new();
                    for _ in 0..*args {
                        let arg = *stack
                            .last()
                            .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;
                        arg_vs.push(arg);
                    }
                    arg_vs.reverse();

                    let call = builder.ins().call(func, &arg_vs);
                    let result = builder.inst_results(call)[0];

                    stack.push(result);
                }
                Token::Noop => {}
            }
        }

        let read_ret = stack
            .pop()
            .ok_or(JitError::CompileInternal("RPN stack exhausted"))?;
        builder.ins().return_(&[read_ret]);
        builder.finalize();

        self.module.define_function(id, &mut self.module_ctx)?;

        self.module.clear_context(&mut self.module_ctx);
        self.module.finalize_definitions()?;

        let code = self.module.get_finalized_function(id);

        let func = unsafe {
            std::mem::transmute::<_, fn(f32, f32, f32, f32, f32, f32, &mut f32, &mut f32) -> f32>(
                code,
            )
        };

        Ok(func)
    }

    /// Free the functions built by this [`Compiler`]
    ///
    /// SAFETY:
    /// - None of the function pointers returned from this compiler can run
    ///   at the moment this function is called or ever called again.
    pub unsafe fn free_memory(self) {
        self.module.free_memory();
    }
}

/// Default names for [ir::LibCall]s. A function by this name is imported into the object as
/// part of the translation of a [ir::ExternalName::LibCall] variant.
fn default_libcall_names() -> Box<dyn Fn(ir::LibCall) -> String + Send + Sync> {
    Box::new(move |libcall| match libcall {
        ir::LibCall::Probestack => "__cranelift_probestack".to_owned(),
        ir::LibCall::CeilF32 => "ceilf".to_owned(),
        ir::LibCall::CeilF64 => "ceil".to_owned(),
        ir::LibCall::FloorF32 => "floorf".to_owned(),
        ir::LibCall::FloorF64 => "floor".to_owned(),
        ir::LibCall::TruncF32 => "truncf".to_owned(),
        ir::LibCall::TruncF64 => "trunc".to_owned(),
        ir::LibCall::NearestF32 => "nearbyintf".to_owned(),
        ir::LibCall::NearestF64 => "nearbyint".to_owned(),
        ir::LibCall::FmaF32 => "fmaf".to_owned(),
        ir::LibCall::FmaF64 => "fma".to_owned(),
        ir::LibCall::Memcpy => "memcpy".to_owned(),
        ir::LibCall::Memset => "memset".to_owned(),
        ir::LibCall::Memmove => "memmove".to_owned(),
        ir::LibCall::Memcmp => "memcmp".to_owned(),

        ir::LibCall::ElfTlsGetAddr => "__tls_get_addr".to_owned(),
        ir::LibCall::ElfTlsGetOffset => "__tls_get_offset".to_owned(),
        ir::LibCall::X86Pshufb => "__cranelift_x86_pshufb".to_owned(),
    })
}
