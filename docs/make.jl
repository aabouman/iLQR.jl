using iLQR
using Documenter

Documenter.makedocs(
                    root="./",
                    source="src",
                    build="build",
                    clean=true,
                    doctest=false,
                    modules=Module[iLQR],
                    repo="",
                    highlightsig=true,
                    sitename="iLQR_Documentation",
                    expandfirst=[],
                    pages=[
                           "Index"=>"index.md"
                           ]
                    )
