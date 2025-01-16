use crate::dispatcher::Dispatcher;
use crate::target::Target;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::ItemFn;

pub(crate) fn make_multiversioned_fn(
    _attr: TokenStream,
    func: ItemFn,
) -> Result<TokenStream, syn::Error> {
    let targets = vec![
        Target::new("x86_64", &["avx2", "fma"]),
        Target::new("x86_64", &["sse4.2"]),
        Target::new("x86", &["avx2", "fma"]),
        Target::new("x86", &["sse4.2"]),
        Target::new("x86", &["sse2"]),
        Target::new("aarch64", &["neon"]),
    ];
    // let default_targets = [
    //     // "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    //     "x86_64+avx2+fma",
    //     "x86_64+sse4.2",
    //     // "x86+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    //     "x86+avx2+fma",
    //     "x86+sse4.2",
    //     "x86+sse2",
    //     "aarch64+neon",
    //     // "arm+neon",
    //     // "mips+msa",
    //     // "mips64+msa",
    //     // "powerpc+vsx",
    //     // "powerpc+altivec",
    //     // "powerpc64+vsx",
    //     // "powerpc64+altivec",
    // ];

    let inner_attrs = Vec::new();

    Ok(Dispatcher {
        targets,
        func,
        inner_attrs,
    }
    .to_token_stream())
}
