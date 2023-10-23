# step43

using Judo
using Random
using Plots
Random.seed!(0)

const lr = 0.2
const iters = 10000



function main()
    x = rand(100)
    y = sin.(2pi * x) + rand(100)

    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * randn(I, H))
    b1 = Variable(zeros(H)')          # b 是行向量
    W2 = Variable(0.01 * randn(H, O))
    b2 = Variable(zeros(O)')

    # 内部函数可以看到外部函数定义的变量，所以使用内部函数要非常小心
    function predict(x) 
        tmp = affine(x, W1, b1)
        tmp = sigmoid(tmp)
        return affine(tmp, W2, b2)
    end

    for i in 1:iters
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        cleargrad!(W1)
        cleargrad!(b1)
        cleargrad!(W2)
        cleargrad!(b2)
        @inference gradient!(loss)

        W1.value -= lr * W1.grad.value
        b1.value -= lr * b1.grad.value
        W2.value -= lr * W2.grad.value
        b2.value -= lr * b2.grad.value
        if (i-1) % 1000 == 0
            println(loss)
        end
    end

    # Plot
    plt = scatter(x, y, marksize=10,xlabel="x", ylabel="y", legend=false)
    t = 0:0.01:1
    y_pred = predict(t)
    plot!(plt, t, y_pred.value, color=:red)
    display(plt)
end

@time main();