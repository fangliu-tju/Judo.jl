# step09

mutable struct Variable
    value
    grad
    creator
end
Variable(data) = Variable(data, nothing, nothing)
Variable() = Variable(nothing, nothing, nothing)

abstract type Func end  
function (f::Func)(fun::Function, input::Variable)
    x = input.value       
    y = fun(x) 
    output = Variable(y)           
    f.input = input       
    f.output = output          
    setcreator!(output, f)
    #output.creator = f   
    return output      
end


setcreator!(v::Variable, func::Func) = v.creator = func

# Square
mutable struct Square <: Func 
    input  
    output 
end

# 构造函数
Square() = Square(nothing, nothing) 

# 求值
_square(x) = Square()(x) do t  # 定义了一个普通函数, 注意这里 `x` 和 `t` 的不同含义，
    t.^2                       # `x` 是 `Square` 的实参， `t` 是匿名函数的形参， 
end                            # 可以是任意合法字符， 比如后面的函数就用 `x` 表示！！！

# 为已有函数创建新方法
# Julia 的函数是一般函数， 方法才是根据参数类型和参数数量创建的， 这里针对新数据类型创建了新方法
Base.Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Variable, ::Val{2}) = _square(x)

# 求局部导数和上游传来导数的乘积
function ∇(f::Square, gy)  
    x_value = f.input.value 
    2 .* x_value .* gy      
end

# Exp
mutable struct Exp <: Func
    input
    output
end

# 构造函数
Exp() = Exp(nothing, nothing)

# 求值
_exp(x) = Exp()(x) do x       # 定义的函数以 `_` 开头， 提示它是内部函数， 不直接使用
    exp.(x)
end

# 添加新方法
# 这个新创建的方法好理解， 上一个复杂一点， 因为底层涉及到了非 `Julia` 实现
Base.exp(x::Variable) = _exp(x) 

# 求导
function ∇(f::Exp, gy)          
    x_value = f.input.value
    exp.(x_value) .* gy
end

function gradient!(v::Variable)
    isnothing(v.grad) && (v.grad = ones(eltype(v.value), size(v.value))) # 同类型同形状
    funcs = Func[] 
    f = v.creator     
    isnothing(f) && return 
    push!(funcs, f)        
    while !isempty(funcs)
        f = pop!(funcs)    
        x, y = f.input, f.output  
        x.grad = ∇(f, y.grad)     
        !isnothing(x.creator) && push!(funcs, x.creator) 
    end
end

# main
x = Variable([0.5])
a = x.^2
b = exp(a)
y = b.^2

gradient!(y)
println(x.grad)

x = Variable(0.5)
y = (exp(x.^2)).^2  # 连续调用

gradient!(y)
println(x.grad)