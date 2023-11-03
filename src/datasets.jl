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
                data, label = _prepare($(esc(name)),train)
                new(train,transform,target_transform,data,label)
            end
        end
    end
end

@createdata Spiral
_prepare(::Type{Spiral},train) = get_spiral(train=train)

function get_spiral(;train=true)
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
    x = [x[i,j] for i in indices, j in 1:input_dim ]
    t = t[indices]
    #Base.permutecols!!(x',indices)
    return x, t
end

@createdata MNIST 
function _prepare(::Type{MNIST},train)
    url = "http://yann.lecun.com/exdb/mnist/" 
    train_files = Dict("target"=>"train-images-idx3-ubyte.gz",
                        "label"=>"train-labels-idx1-ubyte.gz")
    test_files = Dict("target"=>"t10k-images-idx3-ubyte.gz",
                      "label"=>"t10k-labels-idx1-ubyte.gz")
    
    files = train ? train_files : test_files
    data_path = get_file(url * files["target"])
    label_path = get_file(url * files["label"])
    return _load_data(data_path), _load_label(label_path)
end

function _load_label(filepath)
    GZip.open(filepath) do io
        skip(io, 8)
        Int.(read(io))
    end
end

function _load_data(filepath)
    total_items, nrows, ncols, data = 
    GZip.open(filepath) do io
        skip(io, 4)
        return (
        Int(bswap(read(io, UInt32))),
        Int(bswap(read(io, UInt32))),
        Int(bswap(read(io, UInt32))),
        Float64.(read(io))
        )
    end
    reshape(data,nrows*ncols,total_items)'
end
