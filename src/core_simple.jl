# core

# 全局常量， 根据要不要对函数求导， 设置 `true` 或 `false`, 默认 `true`
const enable_grad = Ref(true) 

# 只做推理时不求导
macro inference(ex)
    quote
        enable_grad[] = false
        local val = Base.@__tryfinally($(esc(ex)),
        enable_grad[] = true);  
        val
    end
end

# 核心数据结构， 使变量具备自动微分属性
mutable struct Variable
    value       # 变量的取值
    grad        # 上游变量对该变量的导数值
    creator     # 该变量的创建函数
    generation  # 该变量的辈分
    name        # 该变量的名字
    
    # 内部构造函数， 覆盖默认构造函数
    function Variable(data::AbstractArray; name=nothing)  
        v = new(data)
        v.grad = nothing
        v.creator = nothing
        v.generation = 1
        v.name = name
        return v
    end
end
# 外部构造函数
Variable(data::Number; name=nothing) = Variable([data], name=name)

# 针对新数据类型， 扩展已有函数的方法
Base.ndims(v::Variable) = ndims(v.value)
Base.size(v::Variable) = size(v.value)
Base.size(v::Variable, i) = size(v.value, i)
Base.eltype(v::Variable)  = eltype(v.value)
Base.length(v::Variable) = length(v.value)
Base.show(io::IO, v::Variable) = println(io, "Variable(", v.value, ")")
function Base.show(io::IO, ::MIME"text/plain", v::Variable) 
    println(io, "Variable\n", v.value)
end
Base.convert(::Type{Variable},x::Variable) = x
Base.convert(::Type{Variable},x) = Variable(x)

# 核心数据结构， 使函数具备自动微分属性
abstract type Func end  

# 通过宏创建函数类型， 实现代码复用
macro createfunc(name, arg...) 
    return quote
        mutable struct $(esc(name)) <: Func
            $(arg...)
            inputs
            outputs
            x_shape
            generation
            $(esc(name))($(arg...)) = new($(arg...),nothing,nothing,nothing,nothing)
        end
    end
end

# 实现函数求值， 及在这个过程中建立起函数和变量之间的相互联系
function (f::Func)(fun, inputs...)
    inputs = map(x->convert(Variable,x), inputs) 
    xs = [input.value for input in inputs] 
    ys = fun(xs...)   
    !isa(ys, Tuple) && (ys = tuple(ys))
    outputs = Variable.(ys) 

    if enable_grad[] 

        f.inputs = inputs       
        f.outputs = outputs

        f.generation = mapreduce(x->x.generation, max, inputs) 
        

        for output in outputs 
            setcreator!(output, f) 
        end
    end

    return length(outputs) == 1 ? outputs[1] : outputs        
end

# 由于广播会有很多问题， 调试很麻烦， 所以这里对运算符进行了重写， 不使用广播功能， 可实现需要的功能
# Add
# 1、创建
@createfunc Add
# 2、 求值
_add(x, y) = Add()(x, y) do x, y  
    x .+ y
end                            
# 3、 扩展
Base.:+(x::Variable, y::Variable) = _add(x, y)
Base.:+(x, y::Variable) = _add(x, y) 
Base.:+(x::Variable, y) = _add(x, y) 
# 4、求导
function ∇(f::Add, gy)  
    gy, gy    
end

# Mul
# 1、创建
@createfunc Mul
# 2、求值
_mul(x, y) = Mul()(x, y) do x, y
    x .* y
end
# 3、扩展
Base.:*(x::Variable, y::Variable) = _mul(x, y) 
Base.:*(x::Variable, y) = _mul(x, y) 
Base.:*(x, y::Variable) = _mul(x, y) 
# 4、求导
function ∇(f::Mul, gy)
    x1, x2 = f.inputs
    gy .* x2.value, gy .* x1.value
end
# Neg
# 1、创建
@createfunc Neg
# 2、求值
_neg(x) = Neg()(x) do x
    -x
end
# 3、扩展
Base.:-(x::Variable) = _neg(x)
# 4、求导
∇(f::Neg, gy) = -gy

# Sub
# 1、创建
@createfunc Sub
# 2、求值
_sub(x, y) = Sub()(x, y) do x, y
    x .- y
end
# 3、扩展
Base.:-(x::Variable, y::Variable) = _sub(x, y)
Base.:-(x::Variable, y) = _sub(x, y)
Base.:-(x, y::Variable) = _sub(x, y)
# 4、求导
function ∇(f::Sub, gy) 
    gy, -gy
end

# Div
# 1、创建
@createfunc Div
# 2、求值
_div(x, y) = Div()(x, y) do x, y
    x ./ y
end
# 3、扩展
Base.:/(x::Variable, y::Variable) = _div(x, y)
Base.:/(x::Variable, y) = _div(x, y)
Base.:/(x, y::Variable) = _div(x, y)
# 4、求导
function ∇(f::Div, gy) 
    x1, x2 = f.inputs
    gx1 = gy ./ x2.value
    gx2 = gy .* (-x1.value ./ x2.value.^2)
    return gx1, gx2
end

# Pow
# 1、创建
@createfunc Pow c::Real
# 2、求值
_pow(x, c) = Pow(c)(x) do x
    x .^c
end
# 3、扩展
Base.:^(x::Variable, c)  = _pow(x, c)
# 4、求导
∇(f::Pow, gy) = f.c .* f.inputs[1].value .^(f.c - 1) .* gy

# 求整体导数
function gradient!(v::Variable; retain_grad=false) 
    hasgrad(v) || (v.grad = one.(v.value)) 
    funcs = Func[] 
    seen_set = Set() 
    
    function addfunc(f) 
        if f ∉ seen_set       
            push!(funcs, f)   
            push!(seen_set, f)
            sort!(funcs, lt=(a, b) -> a.generation < b.generation) 
        end
    end

    hascreator(v) || error("Variable MUST be created by some function")
    addfunc(v.creator)     
     
    while !isempty(funcs)
        f = pop!(funcs)    
        gys = [output.grad for output in f.outputs] 
        gxs = ∇(f, gys...) 
        !isa(gxs, Tuple) && (gxs = tuple(gxs)) 
        for (x, gx) in zip(f.inputs, gxs)
            x.grad = (hasgrad(x) ? x.grad : zero(gx)) + gx
            !isnothing(x.creator) && addfunc(x.creator) 
        end
        
        retain_grad || [output.grad = nothing for output in f.outputs]

    end
end

