using Judo
using Documenter

DocMeta.setdocmeta!(Judo, :DocTestSetup, :(using Judo); recursive=true)

makedocs(;
    modules=[Judo],
    authors="Fang Liu <fangliu@tju.edu.cn> and contributors",
    repo="https://github.com/fangliu-tju/Judo.jl/blob/{commit}{path}#{line}",
    sitename="Judo.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fangliu-tju.github.io/Judo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fangliu-tju/Judo.jl",
    devbranch="main",
)
