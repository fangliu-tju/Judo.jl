module Judo
using Random
using GZip
export 
    # variables.jl
    Variable, Parameter, Literal, 
    
    # functions.jl
    broadcastto, sumto, mean_squared_error,
    affine, sigmoid, gradient!, ⋅,
    relu, softmax, softmax_cross_entropy,
    @inference, @createfunc, 

    # layers.jl
    Linear, Layer, @createmodel, MLP,

    # optimizer.jl
    @createopt, setup!, update!, SGD,
    MomentumSGD, 

    # utils.jl
    cleargrad!, cleargrads!, plot_dot_graph,
    params, accuracy, 

    # datasets.jl
    get_spiral, @createdata, Spiral, MNIST,

    # dataloaders.jl
    @createloader, RndDataLoader

include("variables.jl")
include("functions.jl")
include("layers.jl")
include("optimizer.jl")
include("datasets.jl")
include("dataloaders.jl")
include("utils.jl")

end
