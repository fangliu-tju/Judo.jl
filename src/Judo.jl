module Judo

export 
    # core
    Variable, gradient!, cleargrad!,
    @inference, @createfunc

include("core.jl")

end
