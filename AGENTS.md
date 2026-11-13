# Rust Project Rules
- Always use idiomatic Rust (clippy-compliant).
- Prefer `cloned()` over `clone()` where applicable on iterators.
- Use crate-specific `error` module and `phyne::Result` type for error handling.
- Ensure all public functions have explicit documentation and types.
- Check build errors using `cargo check` before finalizing.
- Always use functions defined for DMatrix/DVector (https://docs.rs/nalgebra/latest/nalgebra/base/type.DMatrix.html) for any mathematical operations on such objects.
- Always iterate through tree nodes and sites second when dealing with trees/MSAs/MASAs.
- Prefer docstrings (starting with ///) over regular comments (starting with //) even if the method is private.