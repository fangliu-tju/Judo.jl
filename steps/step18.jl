# step18
# Julia 垃圾回收机制， 用的是 `mark and sweep`, 所以不用关心循环引用问题。 

# Config
const enable_grad = Ref(true) # 全局常数， 控制求值的同时建不建立联系

# 定义一个宏， 当做推理运算时， 不建立函数与变量之间的相互联系
macro inference(ex)
    quote
        enable_grad[] = false
        local val = Base.@__tryfinally($(esc(ex)),
        enable_grad[] = true);  # 当 `ex` 出错时， `enable_grad` 也设置为 `true` 
        val
    end
end

mutable struct Variable
    value
    grad
    creator
    generation  
end
Variable(data) = Variable(data, nothing, nothing, 1) 
Variable() = Variable(nothing, nothing, nothing, 1)

abstract type Func end  

function (f::Func)(fun, inputs...)
    xs = [input.value for input in inputs]   
    ys = fun(xs...)   
    !isa(ys, Tuple) && (ys = tuple(ys))
    outputs = Variable.(ys) 

    # 是不是进行训练， 只有在训练时， 才需要建立起函数和变量之间的联系
    if enable_grad[] # 需要计算梯度时， 执行下面的操作
        # 从函数的视角， 建立起与变量的联系
        f.inputs = inputs       
        f.outputs = outputs
        # 设置函数的辈分
        f.generation = mapreduce(x->x.generation, max, inputs) 
        
        # 从变量的视角， 建立起与函数的联系
        for output in outputs 
            setcreator!(output, f) 
        end
    end

    return length(outputs) == 1 ? outputs[1] : outputs        
end

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.generation + 1 
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
Base.Broadcast.broadcasted(::typeof(+), x1::Variable, x2::Variable) = _add(x1, x2) 

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
Base.Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Variable, ::Val{2}) = _square(x)

# Local Gradient
function ∇(f::Square, gy)  
    x_value = f.inputs[1].value 
    2 .* x_value .* gy      
end

# 求整体导数
function gradient!(v::Variable; retain_grad=false) # 新增关键字参数 `retain_grad`
    isnothing(v.grad) && (v.grad = one.(v.value)) # 改写， 使代码更简洁
    funcs = Func[] 
    seen_set = Set() 
    
    function addfunc(f) 
        if f ∉ seen_set       
            push!(funcs, f)   
            push!(seen_set, f)
            sort!(funcs, lt=(a, b) -> a.generation < b.generation) 
        end
    end

    isnothing(v.creator) && return 
    addfunc(v.creator)     
     
    while !isempty(funcs)
        f = pop!(funcs)    
        gys = [output.grad for output in f.outputs] 
        gxs = ∇(f, gys...) 
        !isa(gxs, Tuple) && (gxs = tuple(gxs)) 
        for (x, gx) in zip(f.inputs, gxs)
            x.grad = (isnothing(x.grad) ? zero(gx) : x.grad) + gx
            !isnothing(x.creator) && addfunc(x.creator) 
        end
        # 决定是否保留中间导数， 不管保留与否， 变量与函数之间的相互联系还是建立的
        retain_grad || [output.grad = nothing for output in f.outputs]

    end
end


cleargrad!(v::Variable) = (v.grad = nothing)

# main
x = Variable([2.0])
a = x.^2
y = a.^2 .+ a.^2
gradient!(y,retain_grad=true)  # 默认不保留中间结果的梯度

@show a.grad
@show x.grad

@inference y = x.^2 # 推理计算， 只对模型给出的结果感兴趣， 不用来进行模型训练
@show y.creator

y = x.^2         # 普通计算， 默认是要用来训练的
@show y.creator;
