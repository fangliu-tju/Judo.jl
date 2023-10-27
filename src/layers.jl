# Layer
# 新定义一个数据类型， 层, 用于管理训练参数
abstract type Layer end

# 应用模型对输入值进行求值计算， 也就是推理
function (l::Layer)(inputs...)
    outputs = evaluation(l, inputs...)
    isa(outputs, Tuple) || (outputs = tuple(outputs))
    l.inputs = inputs
    l.outputs = outputs
    return length(outputs) == 1 ? outputs[1] : outputs
end

# 创建模型的宏， 模型也是层， 但可以有1~N层
macro createmodel(name)
    return quote
        struct $(esc(name)) <: Layer
            _params
            _items
            function $(esc(name))()
                params = Set{Symbol}()
                items = Dict{Symbol, Any}()
                new(params, items)
            end
        end
    end
end

# Linear 是一个基础模型， 复杂模型都由它组成
@createmodel Linear
# 外部构造函数
function Linear(out_size; in_size=nothing,nobias=false)
    l = Linear()
    l.in_size = in_size
    l.out_size = out_size
    l.W = Parameter(NaN,name="W")
    isnothing(l.in_size) || initW!(l)
    l.b = nobias ? nothing : Parameter(zeros(out_size)',name="b")
    return l
end
# 模型求值核心算法
function evaluation(l::Linear, x)
    if isnothing(l.in_size)
        l.in_size = length(size(x)) == 1 ? 1 : last(size(x))
        initW!(l)
    end
    affine(x, l.W, l.b)
end

# MLP
# 1、创建模型
@createmodel MLP
# 2、设置模型参数
function MLP(fc_output_sizes; activation=sigmoid)
    model = MLP() # 初始化模型
    model.activation = activation # 设置激活函数
    model.layers = Symbol[] # 将线性层存入一个数组， 为了节约内存， 只存层名

    for (i, out_size) in enumerate(fc_output_sizes)
        layer = Linear(out_size) # 各层的内容
        index = Symbol("l$i")    # 各层的名字
        setproperty!(model, index, layer) # 存入各自域中
        push!(model.layers, index) # 按顺序加入层名
    end
    return model
end

# 3、求值， 正向传播
function evaluation(model::MLP, x)
    for index in model.layers[begin:end-1]
        l = getproperty(model, index)
        x = model.activation(l(x))
    end
    l = getproperty(model, model.layers[end])
    l(x)
end
