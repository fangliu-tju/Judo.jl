# core

# Config
const enable_grad = Ref(true) 


macro inference(ex)
    quote
        enable_grad[] = false
        local val = Base.@__tryfinally($(esc(ex)),
        enable_grad[] = true);  
        val
    end
end

mutable struct Variable
    value
    grad
    creator
    generation 
    name       
    
    function Variable(data::AbstractArray; name=nothing)  
        v = new(data)
        v.grad = nothing
        v.creator = nothing
        v.generation = 1
        v.name = name
        return v
    end
end
Variable(data::Number; name=nothing) = Variable([data], name=name)


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
Base.iterate(x::Variable) = (x, nothing)
Base.iterate(x::Variable, ::Any) = nothing
abstract type Func end  


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

function setcreator!(v::Variable, func::Func)
    v.creator = func
    v.generation = func.generation + 1 
end

# 由于广播会有很多问题， 调试很麻烦， 所以这里对运算符进行了重写， 不使用广播功能， 可实现需要的功能
# Add
#创建
@createfunc Add
# 求值
_add(x, y) = Add()(x, y) do x, y  
    x .+ y
end                            
# 为已有函数创建新方法
Base.:+(x::Variable, y::Variable) = _add(x, y)
Base.:+(x, y::Variable) = _add(x, y) 
Base.:+(x::Variable, y) = _add(x, y) 
# 求局部导数
function ∇(f::Add, gy)  
    gy, gy    
end

# Mul
# 创建
@createfunc Mul
# 求值
_mul(x, y) = Mul()(x, y) do x, y
    x .* y
end
# 为已有函数创建新方法
Base.:*(x::Variable, y::Variable) = _mul(x, y) 
Base.:*(x::Variable, y) = _mul(x, y) 
Base.:*(x, y::Variable) = _mul(x, y) 
# 求局部导数
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
    isnothing(v.grad) && (v.grad = one.(v.value)) 
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
        
        retain_grad || [output.grad = nothing for output in f.outputs]

    end
end

cleargrad!(v::Variable) = (v.grad = nothing)
