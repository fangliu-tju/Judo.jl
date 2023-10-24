# step44

using Judo
using Random
using Plots
Random.seed!(0)

x = rand(100)
y = sin.(2pi * x) + rand(100)

l1 = Linear(10)
l2 = Linear(1)
function predict(x)
    tmp = l1(x)
    tmp = sigmoid(tmp)
    tmp = l2(tmp)
    return tmp
end

const lr = 0.2
const iters = 10000

for i in 1:iters
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    cleargrads!(l1)
    cleargrads!(l2)
    backward!(loss)

    for l in [l1, l2]
        l.W.data -= lr * l.W.grad.data
        l.b.data -= lr * l.b.grad.data
    end
    
    if (i-1) % 1000 == 0
        println(loss)
    end
end
# Plot
plt = scatter(x, y, marksize=10,xlabel="x", ylabel="y", legend=false)
t = 0:0.01:1
y_pred = predict(t)
plot!(plt, t, y_pred.data, color=:red)