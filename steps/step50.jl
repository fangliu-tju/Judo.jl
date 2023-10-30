# step50

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
    train_set = Spiral(train=true)  # 训练集
    test_set = Spiral(train=false)  # 测试集
    train_loader = RndDataLoader(train_set, batch_size)
    test_loader = RndDataLoader(test_set, batch_size, shuffle=false)

    model = MLP((hidden_size, 3)) # 建立神经网络模型
    opt = SGD(lr=lr)              # 建立优化算法
    setup!(opt, model)            # 将优化算法与神经网络模型相结合

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in 1:max_epoch       # 进行` max_epoch` 次训练
        sum_loss, sum_acc = 0, 0
        for (x,t) in train_loader
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            cleargrads!(model)
            @inference gradient!(loss)
            update!(opt)

            sum_loss += loss.value[1] * length(t)
            sum_acc += acc * length(t)
        end
        
        # Print loss and accuracy every epoch
        avg_loss = sum_loss / size(train_set, 1) # 整个数据的平均损失
        avg_acc = sum_acc / size(train_set, 1)
        @printf "epoch: %d\n train loss: %.4f, accuracy: %.4f\n" epoch  avg_loss avg_acc
        push!(train_loss, avg_loss)
        push!(train_acc, avg_acc)

        sum_loss, sum_acc = 0, 0
        for (x,t) in test_loader
            @inference y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss += loss.value[1] * length(t)
            sum_acc += acc * length(t)
        end
        avg_loss = sum_loss / size(test_set, 1) # 整个数据的平均损失
        avg_acc = sum_acc / size(test_set, 1)
        @printf "epoch: %d\n test loss: %.4f, accuracy: %.4f\n" epoch  avg_loss avg_acc
        push!(test_loss, avg_loss)
        push!(test_acc, avg_acc)
    end
    plt1 = plot([train_loss test_loss],label=["train" "test"],xlabel="epoch",ylabel="loss")
    plt2 = plot([train_acc test_acc],label=["train" "test"],xlabel="epoch",ylabel="accuracy")
    plt = plot(plt1,plt2,layout=(1,2))
    display(plt)
end

@time main();