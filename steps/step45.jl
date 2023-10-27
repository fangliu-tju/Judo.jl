# step45

using Judo
using Random
using Plots
Random.seed!(0)

x = rand(100)
y = sin.(2pi * x) + rand(100)

const lr = 0.2
const iters = 10000
hidden_size = 10

# 创建模型
@createmodel TwoLayerNet
# 设置模型
function TwoLayerNet(hidden_size, out_size)
    model = TwoLayerNet()
    model.l1 = Linear(hidden_size)
    model.l2 = Linear(out_size)
    model 
end
# 定义模型核心算法
# 由于 `evaluation` 没有输出， 所以这里要加上名称空间
function Judo.evaluation(model::TwoLayerNet, x) 
    y = sigmoid(model.l1(x))
    model.l2(y)
end

# main
function main()
    model = TwoLayerNet(hidden_size, 1)
    #model = MLP((hidden_size,1)) # 更一般的实现

    for i in 1:iters
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        cleargrads!(model)
        @inference gradient!(loss)

        for p in params(model)
            p.value -= lr * p.grad.value
        end
    
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