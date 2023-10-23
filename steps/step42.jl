# step42

using Judo
using Random
using Plots
Random.seed!(0)

const lr = 0.1
const iters = 100

#=
function mean_squared_error(x1, x2)
    diff = x1 - x2
    sum(diff^2)/length(diff)
end
=#

function main() 
    x = rand(100)
    y = 5 .+ 2x + rand(100)

    W = Variable(zeros(1))
    b = Variable(zeros(1))

    predict(x) = x ⋅ W + b

    for _ in 1:iters
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(W)
        cleargrad!(b)

        @inference gradient!(loss)

        W.value -= lr * W.grad.value
        b.value -= lr * b.grad.value
        @show W, b, loss
    end

    # Plot
    plt = scatter(x, y, marksize=10,xlabel="x",ylabel="y",legend=false)

    t = 0:0.01:1
    y_pred = predict(t)
    plot!(plt, t, y_pred.value, color=:red)
    display(plt)
end

@time main();