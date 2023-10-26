# step45

using Judo
using Random
using Plots
Random.seed!(0)

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
x = Variable(randn(5, 10), name="x")
model = TwoLayerNet(100, 10)
plot_dot_graph(model, x, file="images/twolayernet.png")

#=
x = rand(100)
y = sin.(2pi * x) + rand(100)

model = MLP((10,1))

const lr = 0.2
const iters = 10000

for i in 1:iters
    y_pred = model(x)
    loss = mean_squared_error(y, y_pred)

    cleargrads!(model)
    backward!(loss)

    for l in model.layers
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
y_pred = model(t)
plot!(plt, t, y_pred.data, color=:red)
=#