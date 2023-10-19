# step13

mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun, inputs...) # 函数更具一般性
    xs = [input.value for input in inputs] # 遍历数组      
    ys = fun(xs...)   # 在调用普通函数时， 将数组展开成单个参数
    !isa(ys, Tuple) && (ys = tuple(ys)) # 返回值如果不是元组， 则转换成元组
    outputs = Variable.(ys) # `Variable` 是针对单个参数的， 对于数组要用到 `.` 运算符
    for output in outputs          
        setcreator!(output, f) # 每个输出都使用了相同的运算
    end         
    f.inputs = inputs       
    f.outputs = outputs
    return length(outputs) == 1 ? outputs[1] : outputs # 只有一个值时， 返回值          
end

setcreator!(v::Variable, func::Func) = v.creator = func

# Add
mutable struct Add <: Func 
    inputs  
    outputs 
end

# 构造函数
Add() = Add(nothing, nothing) 

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
end

# Constructor
Square() = Square(nothing, nothing)

# Evaluation
_square(x) = Square()(x) do x
    x.^2
end

# Dispatch
Base.:^(x::Variable, c) = _square(x)

# Gradient
function ∇(f::Square, gy)  
    x_value = f.inputs[1].value 
    2 .* x_value .* gy      
end

# 求整体导数
function gradient!(v::Variable)
    isnothing(v.grad) && (v.grad = one.(v.value))
    funcs = Func[] 
    f = v.creator     
    isnothing(f) && return 
    push!(funcs, f)        
    while !isempty(funcs)
        f = pop!(funcs)    
        gys = [output.grad for output in f.outputs] # 依次取出多个输出的梯度 
        gxs = ∇(f, gys...) # 函数输出是多个时， 需要定义相应的梯度函数
        !isa(gxs, Tuple) && (gxs = tuple(gxs)) # 转换成元组， 针对可变长度参数
        for (x, gx) in zip(f.inputs, gxs)
            x.grad = gx
            !isnothing(x.creator) && push!(funcs, x.creator) 
        end
    end
end

# main
x = Variable([2.0, 3.0])
y = Variable([3.0])
z = x^2 + y^2
gradient!(z)
println(z.value)
println(x.grad)
println(y.grad)