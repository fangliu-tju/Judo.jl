module Judo

export 
    # core
    Variable, Parameter, Literal, Linear,
    Layer,
    gradient!, ⋅,
    @inference, @createfunc, @createmodel, 
    # functions
    broadcastto, sumto, mean_squared_error,
    affine, sigmoid,
    # utils
    cleargrad!, cleargrads!, plot_dot_graph,
    params

include("core.jl")
include("functions.jl")
include("utils.jl")

end
