# step27

include("../src/core_simple.jl")
include("../src/utils.jl")

function my_sin(x, threshold=1e-4)
    y = 0
    for i in 0:100000
        c = (-1)^i / factorial(big(2i + 1))
        t = c * x^(2i + 1)
        y = y + t
        all(abs.(t.value) .< threshold) && break
    end
    return y
end 

function main()
    x = Variable([pi / 4], name="x")
    y = my_sin(x, 1e-150)
    y.name="my_sin"
    @show y
    gradient!(y,retain_grad=true)
    @show y.value
    @show x.grad
    plot_dot_graph(y, verbose=false, file="images/sin_1e-150.png")
end

@time main()