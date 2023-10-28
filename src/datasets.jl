# datasets

# 定义抽象数据集类型
abstract type Dataset end

# 创建数据集类型
macro createdata(name)
    return quote
        struct $(esc(name)) <: Dataset
            train
            transform
            target_transform
            data
            label
            function $(esc(name))(;train=true, transform=identity, target_transform=identity)
                data, label = prepare($(esc(name)),train)
                new(train,transform,target_transform,data,label)
            end
        end
    end
end

@createdata Spiral
function prepare(::Type{Spiral},train=false)
    seed = train ? 1984 : 2020
    Random.seed!(seed)
           
    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = zeros(data_size, input_dim)
    t = zeros(Int, data_size)
           
    for j in 1:num_class
        for i in 1:num_data
            rate = (i-1) / num_data
            radius = 1.0 * rate
            theta = (j-1) * 4.0 + 4.0 * rate + 0.2 * randn()
            ix = num_data * (j-1) + i
            x[ix,:] = [radius * sin(theta), radius * cos(theta)]
            t[ix] = j
        end
    end
    indices = randperm(data_size)
    t = t[indices]
    Base.permutecols!!(x',indices)
    return x, t
end
