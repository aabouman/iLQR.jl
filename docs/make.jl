using Documenter
using iLQR


Documenter.makedocs(
    sitename = "iLQR.jl",
    repo = "https://github.com/aabouman/iLQR.jl",
    pages = [
        "Home" => "index.md",
        # "Documentation" => "documentation.md",
        # "Tutorial" => "tutorial.md"
    ]
)
deploydocs(
    repo = "https://github.com/aabouman/iLQR.jl",
)

# deploydocs(
#     repo = "github.com/m3g/PDBTools.git",
#     target = "build",
#     branch = "gh-pages",
#     # versions = ["stable" => "v^", "v#.#" ],
# )
