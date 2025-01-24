use cranelift_codegen::{settings::SetError, CodegenError};
use cranelift_module::ModuleError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum JitError {
    #[error("expression cannot be parsed")]
    ParseError(#[from] meval::ParseError),
    #[error("RPN cannot be constructed")]
    RpnConstruction(#[from] meval::RPNError),
    #[error("unknown RPN token `{0}`")]
    ParseUnknownToken(String),
    #[error("unknown variable name `{0}`")]
    ParseUnknownVariable(String),
    #[error("unknown binary operation `{0}`")]
    ParseUnknownBinop(String),
    #[error("unknown unary operation `{0}`")]
    ParseUnknownUnop(String),
    #[error("unknown function call `{0}`")]
    ParseUnknownFunc(String),
    #[error("function not present in library `{0}`")]
    CompileUknownFunc(String),
    #[error("internal error `{0}`")]
    CompileInternal(&'static str),
    #[error("couldn't set cranelift setting: {0}")]
    CraneliftSetting(#[from] SetError),
    #[error("module operation failed: {0}")]
    CraneliftModule(#[from] ModuleError),
    #[error("host architecture not supported: {0}")]
    CraneliftHostUnsupported(&'static str),
    #[error("codegen error: {0}")]
    CraneliftCodegenError(#[from] CodegenError),
}
