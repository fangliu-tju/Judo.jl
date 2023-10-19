# step16

mutable struct Variable
    value
    grad
    creator
    generation  # 增加变量的辈分关系
end
Variable(data) = Variable(data, nothing, nothing, 1) # 从第一代起算
Variable() = Variable(nothing, nothing, nothing, 1)

abstract type Func end  
#可调用类型调用的实现， 使用抽象数据类型， 可以实现代码复用
function (f::Func)(fun, inputs...)
    xs = [input.value for input in inputs]   
    ys = fun(xs...) # 只有这一步是和具体是哪种计算有关  
    !isa(ys, Tuple) && (ys = tuple(ys))
    outputs = Variable.(ys) # 这时候的输出只含有`值`的信息
    f.inputs = inputs       # 设置函数与变量的联系，下同
    f.outputs = outputs
    # 设置函数的辈分
    f.generation = mapreduce(x->x.generation, max, inputs) # 取出输入变量中辈分最低的值
    for output in outputs 
        # 补充输出的创造者和辈分信息         
        setcreator!(output, f) # 输出的辈分根据函数的辈分计算， 所以要先计算函数的辈分
    end         
    return length(outputs) == 1 ? outputs[1] : outputs        
end

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.generation + 1 # 输出值比创造者低一辈
end

# Add
mutable struct Add <: Func 
    inputs  
    outputs
    generation 
end

# 构造函数
Add() = Add(nothing, nothing, nothing) 

# 求值
_add(x1, x2) = Add()(x1, x2) do x1, x2  
    x1 .+ x2
end                            

# 为已有函数创建新方法
Base.:+(x1::Variable, x2::Variable) = _add(x1, x2) 

# 求局部导数
function ∇(f::Add, gy)  
    gy, gy    
end

# Square
mutable struct Square <: Func
    inputs
    outputs
    generation
end

# Constructor
Square() = Square(nothing, nothing, nothing)

# Evaluation
_square(x) = Square()(x) do x
    x.^2
end

# Dispatch
Base.:^(x::Variable, c) = _square(x)

# Local Gradient
function ∇(f::Square, gy)  
    x_value = f.inputs[1].value 
    2 .* x_value .* gy      
end

# 求整体导数
function gradient!(v::Variable)
    isnothing(v.grad) && (v.grad = one.(v.value)) 
    funcs = Func[] 
    seen_set = Set() # 防止同一个输出变量的创造者被多次添加
    # 定义一个内部函数， 用于形成有序的函数列表， 该函数使用了父函数的 `funcs` 和 `seen_set`
    function addfunc(f) 
        if f ∉ seen_set       # 函数首次出现
            push!(funcs, f)   # 压入函数列表
            push!(seen_set, f)# 压入函数集合， 这个集只增不减
            sort!(funcs, lt=(a, b) -> a.generation < b.generation) # 根据函数的辈分排序
        end
    end

    isnothing(v.creator) && return # 对无创造者的变量， 也可以求导， 不加这句就会报错
    addfunc(v.creator)     # 添加输入变量的创造者， 允许是 `nothing`
     
    while !isempty(funcs)
        f = pop!(funcs)    
        gys = [output.grad for output in f.outputs] # 依次取出多个输出的梯度 
        gxs = ∇(f, gys...) # 函数输出是多个时， 需要定义相应的梯度函数
        !isa(gxs, Tuple) && (gxs = tuple(gxs)) # 转换成元组， 针对可变长度参数
        for (x, gx) in zip(f.inputs, gxs)
            if isnothing(x.grad)
                x.grad = gx      # 初次使用变量的情况
            else
                x.grad = x.grad + gx # 重复使用变量的情况
            end
            !isnothing(x.creator) && addfunc(x.creator) 
        end
    end
end

# 重复利用变量时， 需要先清除变量在之前计算中得到的梯度
cleargrad!(v::Variable) = (v.grad = nothing)

# main
x = Variable([2.0])
a = x^2
y = a^2 + a^2
gradient!(y)

println(y.value)
println(x.grad)
