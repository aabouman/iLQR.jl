using Documenter, iLQR

Documenter.makedocs(
    sitename = "iLQR.jl",
    # repo = "https://github.com/aabouman/iLQR.jl",
    # modules = [iLQR],
    pages = [
        "Home" => "index.md",
        # "Documentation" => "documentation.md",
        # "Tutorial" => "tutorial.md"
    ]
)

deploydocs(;
    repo="https://github.com/aabouman/iLQR.jl",
)
