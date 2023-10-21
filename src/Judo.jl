module Judo

export 
    # core
    Variable, gradient!, cleargrad!,
    @inference, @createfunc,
    # utils
    plot_dot_graph

include("core.jl")
include("functions.jl")
include("utils.jl")

end
