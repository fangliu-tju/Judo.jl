# step34

using Judo 
using Plots
#=
x = Variable([1.0])
y = sin(x)
gradient!(y)

for _ in 1:3
    gx = x.grad
    cleargrad!(x)
    gradient!(gx)
    println(x.grad)
end
=#
function main()
    x = Variable(collect(range(-7, 7, length=200)))
    y = sin(x)
    gradient!(y)
    logs = [vec(y.value)]

    iters = 3
    for _ in 1:iters
        push!(logs, vec(x.grad.value))
        gx = x.grad
        cleargrad!(x)
        gradient!(gx)
    end

    labels = ["y=sin(x)", "y′", "y′′", "y′′′"]
    plt = plot()
    for (i, v) in enumerate(logs)
        plt = plot!(x.value, v, label=labels[i])
    end
    cd("images")
    savefig(plt, "sin")
end

@time main()