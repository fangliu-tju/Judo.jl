# step29

include("../src/core_simple.jl")
include("../src/utils.jl")

f(x) = x^4 - 2 * x^2
f′′(x) = 12 * x^2 - 4 # 二阶导数是手动计算的

function main()
    x = Variable([2.0])
    iters = 9
    for i in 0:iters
        @show i, x.value
        cleargrad!(x)
        y = f(x)
        gradient!(y)
        x.value -= x.grad ./ f′′.(x.value)
    end 
end

@time main()