# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using RadiationDetectorSignals

# Doctest setup
DocMeta.setdocmeta!(
    RadiationDetectorSignals,
    :DocTestSetup,
    :(using RadiationDetectorSignals);
    recursive=true,
)

makedocs(
    sitename = "RadiationDetectorSignals",
    modules = [RadiationDetectorSignals],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://JuliaPhysics.github.io/RadiationDetectorSignals.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/JuliaPhysics/RadiationDetectorSignals.jl.git",
    forcepush = true,
    push_preview = true,
)
