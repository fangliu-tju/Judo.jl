# step46

using Judo
using Random
using Plots
Random.seed!(0)

x = rand(100)
y = sin.(2pi * x) + rand(100)

const lr = 0.2
const iters = 10000
h_size = 10

# main
function main()
    model = MLP((h_size,1), activation=sigmoid) 
    opt = SGD(lr=lr)
    #opt = MomentumSGD(lr=lr)
    setup!(opt, model)   # 这样做的特点是， 模型与优化器分离
                         # 对于同一个模型， 可以比较不同优化器的效果

    for i in 1:iters
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        cleargrads!(model)
        @inference gradient!(loss)

        update!(opt)
    
        if (i-1) % 1000 == 0
            println(loss)
        end
    end
    # Plot
    plt = scatter(x, y, marksize=10,xlabel="x", ylabel="y", legend=false)
    t = 0:0.01:1
    y_pred = model(t)
    plot!(plt, t, y_pred.value, color=:red)
end

@time main()