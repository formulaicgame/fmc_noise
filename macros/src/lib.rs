//! Implementation crate for `multiversion`.
extern crate proc_macro;

mod cfg;
mod dispatcher;
mod multiversion;
mod target;
mod util;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{ItemFn, parse::Nothing, parse_macro_input, punctuated::Punctuated};

#[proc_macro_attribute]
pub fn multiversion(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let func = parse_macro_input!(input as ItemFn);
    match multiversion::make_multiversioned_fn(attr.into(), func) {
        Ok(tokens) => tokens.into_token_stream(),
        Err(err) => err.to_compile_error(),
    }
    .into()
}

#[proc_macro]
pub fn selected_target(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_macro_input!(input as Nothing);
    quote! {
        __multiversion::FEATURES
    }
    .into()
}

#[proc_macro_attribute]
pub fn target_cfg(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = TokenStream::from(attr);
    let input = TokenStream::from(input);
    quote! {
        __multiversion::target_cfg!{ [#attr] #input }
    }
    .into()
}

#[proc_macro_attribute]
pub fn target_cfg_attr(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = TokenStream::from(attr);
    let input = TokenStream::from(input);
    quote! {
        __multiversion::target_cfg_attr!{ [#attr] #input }
    }
    .into()
}

#[proc_macro]
pub fn target_cfg_f(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = TokenStream::from(input);
    quote! {
        __multiversion::target_cfg_f!{ #input }
    }
    .into()
}

#[proc_macro_attribute]
pub fn target_cfg_impl(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let meta = parse_macro_input!(attr with Punctuated::parse_terminated);
    let input = TokenStream::from(input);

    match cfg::transform(meta) {
        Ok(meta) => {
            quote! {
                #[cfg(#meta)]
                #input
            }
        }
        Err(err) => err.to_compile_error(),
    }
    .into()
}

#[proc_macro_attribute]
pub fn target_cfg_attr_impl(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let mut meta = parse_macro_input!(attr with Punctuated::parse_terminated);
    let input = TokenStream::from(input);

    let attr = meta.pop().unwrap();
    match cfg::transform(meta) {
        Ok(meta) => {
            quote! {
                #[cfg_attr(#meta, #attr)]
                #input
            }
        }
        Err(err) => err.to_compile_error(),
    }
    .into()
}

#[proc_macro]
pub fn target_cfg_f_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let meta = parse_macro_input!(input with Punctuated::parse_terminated);

    match cfg::transform(meta) {
        Ok(meta) => {
            quote! {
                cfg!(#meta)
            }
        }
        Err(err) => err.to_compile_error(),
    }
    .into()
}

#[proc_macro]
pub fn match_target(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = TokenStream::from(input);
    quote! {
        __multiversion::match_target!{ #input }
    }
    .into()
}
