use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Attribute, LitStr, parse_quote};
use target_features::Architecture;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub architecture: String,
    pub features: Vec<String>,
}

impl Target {
    pub fn new(arch: &str, features: &[&str]) -> Self {
        let architecture = Architecture::from_str(arch);
        let mut target = target_features::Target::new(architecture);

        for feature in features {
            target = target.with_feature_str(feature);
        }

        let mut features = target
            .features()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        features.sort_unstable();

        Self {
            architecture: arch.to_owned(),
            features,
        }
    }

    pub fn arch(&self) -> &str {
        &self.architecture
    }

    pub fn features(&self) -> &[String] {
        self.features.as_ref()
    }

    pub fn features_string(&self) -> String {
        self.features.join("_").replace('.', "")
    }

    pub fn has_features_specified(&self) -> bool {
        !self.features.is_empty()
    }

    pub fn target_arch(&self) -> Attribute {
        let arch = &self.architecture;
        parse_quote! {
            #[cfg(target_arch = #arch)]
        }
    }

    pub fn target_feature(&self) -> Vec<Attribute> {
        self.features
            .iter()
            .map(|feature| {
                parse_quote! {
                    #[target_feature(enable = #feature)]
                }
            })
            .collect()
    }

    pub fn fn_attrs(&self) -> Vec<Attribute> {
        let mut attrs = self.target_feature();
        attrs.push(self.target_arch());
        attrs
    }

    pub fn features_detected(&self) -> TokenStream {
        let feature = self.features.iter();
        let is_feature_detected =
            format_ident!("is_{}_feature_detected", match self.architecture.as_str() {
                "x86_64" => "x86",
                "risv64" => "riscv",
                f => f,
            });
        quote! {
            true #( && std::arch::#is_feature_detected!(#feature) )*
        }
    }
}

impl ToTokens for Target {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut s = self.architecture.clone();
        for feature in &self.features {
            s.push('+');
            s.push_str(feature);
        }
        LitStr::new(&s, Span::call_site()).to_tokens(tokens);
    }
}

// pub(crate) fn make_target_fn(target: LitStr, func: ItemFn) -> Result<TokenStream> {
//     let target = Target::parse(&target)?;
//     let target_arch = target.target_arch();
//     let target_feature = target.target_feature();
//     Ok(parse_quote! { #target_arch #(#target_feature)* #func })
// }
