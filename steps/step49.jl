# step49

using Judo
using Random
using Printf
using Plots

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

function main()
    train_set = Spiral()  # 创建数据集
    model = MLP((hidden_size, 3)) # 建立神经网络模型
    opt = SGD(lr=lr)              # 建立优化算法
    setup!(opt, model)            # 将优化算法与神经网络模型相结合

    data_size = size(train_set,1) # 获取训练数据长度
    max_iter = data_size ÷ batch_size # 每次训练的循环次数

    for epoch in 1:max_epoch       # 进行` max_epoch` 次训练
        # Shuffle index for data
        index = randperm(data_size) # 对训练数据进行随机排序
        sum_loss = 0 # 总损失

        for i in 1:max_iter # 每次训练的内部循环， 随机取出少量数据进行学习
            batch_index = index[(i-1)*batch_size+1 : i*batch_size]
            batch_x, batch_t = train_set[batch_index]

            y = model(batch_x)
            loss = softmax_cross_entropy(y, batch_t)
            cleargrads!(model)
            @inference gradient!(loss)
            update!(opt) # 模型参数的改变由优化算法决定， 故更新模型参数实际是更新优化模型

            sum_loss += loss.value[1] * batch_size # 片段数据的总损失
        end

        # Print loss every epoch
        avg_loss = sum_loss / data_size # 整个数据的平均损失
        @printf "epoch %d, loss %.2f\n" epoch  avg_loss
    end


    # Plot boundary area the model predict
    h = 0.001
    x_min, y_min = minimum(train_set.data, dims=1) .- 0.1 # 取出输入数据的下限
    x_max, y_max = maximum(train_set.data, dims=1) .+ 0.1 # 取出输入数据的上限
    x1d = x_min : h : x_max  # 分类用的 `x` 值
    y1d = y_min : h : y_max  # 分类用的 `y` 值
    x2d = x1d .* ones(length(y1d))' # 注意： Plots 中的 `x`、`y` 轴与数学上相反
    y2d = ones(length(x1d)) .* y1d'
    X = [vec(x2d) vec(y2d)]
    @inference score = model(X) # 用训练好的模型进行分类推理
    # 取出分类标签
    predict_cls = [argmax(score.value[i,:]) for i in 1:length(x2d)] 
    # 利用分类标签对区域进行划分
    plt = contourf(x1d,y1d,predict_cls,alpha=0.1,ratio=1)
    # 将训练数据的标签用不同形状表示
    shape = map(train_set.label) do x 
        if x == 1
            :circle 
        elseif x == 2
            :star
        else
            :rect 
        end
    end
    # 将训练数据画在分类图上
    scatter!(plt, train_set.data[:,1], train_set.data[:,2], color=train_set.label,shape=shape,leg=false)
    display(plt) # 显示最终结果
end

@time main();