using Documenter
using iLQR


# Documenter.makedocs(
#     sitename = "iLQR.jl",
#     repo = "https://github.com/aabouman/iLQR.jl",
#     pages = [
#         "Home" => "index.md",
#         # "Documentation" => "documentation.md",
#         # "Tutorial" => "tutorial.md"
#     ]
# )
makedocs(
    sitename = "iLQR.jl",
    repo = "https://github.com/aabouman/iLQR.jl",
    pages = [
            "Home" => "index.md",
        ]
)
deploydocs(
    repo = "https://github.com/aabouman/iLQR.jl",
    branch = "gh-pages",
)

# deploydocs(
#     repo = "github.com/m3g/PDBTools.git",
#     target = "build",
#     branch = "gh-pages",
#     # versions = ["stable" => "v^", "v#.#" ],
# )
