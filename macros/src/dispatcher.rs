use crate::{target::Target, util};
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use std::collections::HashMap;
use syn::{
    Attribute, Block, Ident, ItemFn, Result, ReturnType, Signature, Type, Visibility, parse_quote,
    punctuated::Punctuated, token::RArrow,
};

pub(crate) fn feature_fn_name(ident: &Ident, target: Option<&Target>) -> Ident {
    if let Some(target) = target {
        if target.has_features_specified() {
            return Ident::new(
                &format!("{}_{}_version", ident, target.features_string()),
                ident.span(),
            );
        }
    }

    // If this is a default fn, it doesn't have a dedicated static dispatcher
    Ident::new(&format!("{ident}_default_version"), ident.span())
}

pub(crate) struct Dispatcher {
    pub inner_attrs: Vec<Attribute>,
    pub targets: Vec<Target>,
    pub func: ItemFn,
}

impl Dispatcher {
    // Create functions for each target
    fn feature_fns(&self) -> Result<Vec<ItemFn>> {
        let make_block = |target: Option<&Target>| {
            let block = &self.func.block;
            let features = target.map(|t| t.features()).unwrap_or(&[]);
            let features_init = quote! {
                (target_features::CURRENT_TARGET)#(.with_feature_str(#features))*
            };
            let feature_attrs = if let Some(target) = target {
                target.target_feature()
            } else {
                Vec::new()
            };
            let features = if let Some(target) = target {
                let s = target
                    .features()
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                s.join(",")
            } else {
                String::new()
            };
            parse_quote! {
                {
                    #[doc(hidden)] // https://github.com/rust-lang/rust/issues/111415
                    #[allow(unused)]
                    pub mod __multiversion {
                        pub const FEATURES: target_features::Target = #features_init;

                        macro_rules! inherit_target {
                            { $f:item } => { #(#feature_attrs)* $f }
                        }

                        macro_rules! target_cfg {
                            { [$cfg:meta] $($attached:tt)* } => { #[multiversion::target::target_cfg_impl(target_features = #features, $cfg)] $($attached)* };
                        }

                        macro_rules! target_cfg_attr {
                            { [$cfg:meta, $attr:meta] $($attached:tt)* } => { #[multiversion::target::target_cfg_attr_impl(target_features = #features, $cfg, $attr)] $($attached)* };
                        }

                        macro_rules! target_cfg_f {
                            { $cfg:meta } => { multiversion::target::target_cfg_f_impl!(target_features = #features, $cfg) };
                        }

                        macro_rules! match_target {
                            { $($arms:tt)* } => { multiversion::target::match_target_impl!{ #features $($arms)* } }
                        }

                        pub(crate) use inherit_target;
                        pub(crate) use target_cfg;
                        pub(crate) use target_cfg_attr;
                        pub(crate) use target_cfg_f;
                        pub(crate) use match_target;
                    }
                    #block
                }
            }
        };

        let mut fns = Vec::new();
        for target in &self.targets {
            // This function will always be unsafe, regardless of the safety of the multiversioned
            // function.
            //
            // This could accidentally allow unsafe operations to end up in functions that appear
            // safe, but the deny lint should catch it.
            //
            // For now, nest a safe copy in the unsafe version. This is imperfect, but sound.
            //
            // When target_feature 1.1 is available, this function can instead use the original
            // function safety.
            let mut f = ItemFn {
                attrs: self.inner_attrs.clone(),
                vis: Visibility::Inherited,
                sig: Signature {
                    ident: feature_fn_name(&self.func.sig.ident, Some(target)),
                    unsafety: parse_quote! { unsafe },
                    ..self.func.sig.clone()
                },
                block: make_block(Some(target)),
            };
            f.attrs.extend(target.fn_attrs());
            fns.push(f);
        }

        // Create default fn
        let mut attrs = self.inner_attrs.clone();
        attrs.push(parse_quote! { #[inline(always)] });
        let block = make_block(None);
        fns.push(ItemFn {
            attrs,
            vis: self.func.vis.clone(),
            sig: Signature {
                ident: feature_fn_name(&self.func.sig.ident, None),
                ..self.func.sig.clone()
            },
            block,
        });

        Ok(fns)
    }

    fn pointer_dispatcher_fn(&self) -> Result<Block> {
        let feature_detection = {
            let return_if_detected = self.targets.iter().filter_map(|target| {
                if target.has_features_specified() {
                    let target_arch = target.target_arch();
                    let features_detected = target.features_detected();
                    let function = feature_fn_name(&self.func.sig.ident, Some(target));
                    Some(quote! {
                       #target_arch
                       {
                           if #features_detected {
                               return #function
                           }
                       }
                    })
                } else {
                    None
                }
            });
            let default_fn = feature_fn_name(&self.func.sig.ident, None);
            quote! {
                #(#return_if_detected)*
                return #default_fn
            }
        };

        Ok(parse_quote! {
            {
                #feature_detection
            }
        })
    }

    fn create_fn(&self) -> Result<ItemFn> {
        let block = self.pointer_dispatcher_fn()?;

        // If we already know that the current build target supports the best function choice, we
        // can skip dispatching entirely.
        //
        // Here we check for one of two possibilities:
        // * If the globally enabled features (the target-feature or target-cpu codegen options)
        //   already support the highest priority function, skip dispatch entirely and call that
        //   function.
        // * If the current target isn't specified in the multiversioned list at all, we can skip
        //   dispatch entirely and call the default function.
        //
        // In these cases, the default function is called instead.
        let best_targets = self
            .targets
            .iter()
            .rev()
            .map(|t| (t.arch(), t))
            .collect::<HashMap<_, _>>();
        let mut skips = Vec::new();
        for (arch, target) in best_targets.iter() {
            let feature = target.features();
            skips.push(quote! {
                all(target_arch = #arch, #(target_feature = #feature),*)
            });
        }
        let specified_arches = best_targets.keys().collect::<Vec<_>>();
        let call_default = feature_fn_name(&self.func.sig.ident, None);
        let (normalized_signature, _) = util::normalize_signature(&self.func.sig);
        let feature_fns = self.feature_fns()?;
        let mut return_type = util::fn_type_from_signature(&self.func.sig)?;
        return_type.unsafety = Some(syn::token::Unsafe::default());
        Ok(ItemFn {
            attrs: self.func.attrs.clone(),
            vis: self.func.vis.clone(),
            sig: Signature {
                inputs: Punctuated::default(),
                output: ReturnType::Type(RArrow::default(), Box::new(Type::BareFn(return_type))),
                generics: self.func.sig.generics.clone(),
                ..normalized_signature
            },
            block: Box::new(parse_quote! {
                {
                    #(#feature_fns)*

                    #[cfg(any(
                        not(any(#(target_arch = #specified_arches),*)),
                        #(#skips),*
                    ))]
                    { return #call_default }

                    #[cfg(not(any(
                        not(any(#(target_arch = #specified_arches),*)),
                        #(#skips),*
                    )))]
                    #block
                }
            }),
        })
    }
}

impl ToTokens for Dispatcher {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self.create_fn() {
            Ok(val) => val.into_token_stream(),
            Err(err) => err.to_compile_error(),
        })
    }
}
