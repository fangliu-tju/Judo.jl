# step28

include("../src/core_simple.jl")
include("../src/utils.jl")

rosenbrock(x1,x2) = 100 * (x2 - x1^2)^2 + (x1 - 1)^2

function main()
    x1 = Variable([0.0])
    x2 = Variable([2.0])
    y = rosenbrock(x1, x2)
    gradient!(y)
    println(x1.grad, " ",x2.grad)
    lr = 1e-3
    iters = 1000

    for _ in 1:iters
        println(x1.value, x2.value)
        cleargrad!(x1)
        cleargrad!(x2)
        y = rosenbrock(x1, x2)
        gradient!(y)
            #println(x1.grad.data, x2.grad.data)
        x1.value -= lr .* x1.grad
        x2.value -= lr .* x2.grad
    end
end

@time main();