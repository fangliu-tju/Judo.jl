module Judo

export 
    # core
    Variable, Parameter, Literal,
    gradient!, ⋅,
    @inference, @createfunc,
    # functions
    broadcastto, sumto, mean_squared_error,
    affine, sigmoid,
    # utils
    cleargrad!, plot_dot_graph

include("core.jl")
include("functions.jl")
include("utils.jl")

end
