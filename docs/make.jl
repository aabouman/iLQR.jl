using Documenter, iLQR


Documenter.makedocs(
    sitename = "iLQR.jl",
    source  = "src",
    build   = "build",
    clean   = true,
    doctest = true,
    # repo = "https://github.com/aabouman/iLQR.jl",
    modules = [iLQR],
    pages = ["Home" => "index.md",
             "Documentation" => "documentation.md"
             ]
)

deploydocs(;
    repo = "github.com/aabouman/iLQR.jl",
    branch = "gh-pages",
    devurl = "docs",
)
