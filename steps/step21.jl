# step21

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
# 扩展类型转换函数
Base.convert(::Type{Variable},x::Variable) = x
Base.convert(::Type{Variable},x) = Variable(x)

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
    inputs = map(x->convert(Variable,x), inputs) # 预处理， 将参数转换成 `Variable` 类型
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

# ElAdd
#创建
@createfunc ElAdd
# 求值
_eladd(x, y) = ElAdd()(x, y) do x, y  
    x .+ y
end                            
# 为已有函数创建新方法
Base.Broadcast.broadcasted(::typeof(+), x::Variable, y::Variable) = _eladd(x, y)
Base.Broadcast.broadcasted(::typeof(+), x, y::Variable) = _eladd(x, y) # 新增
Base.Broadcast.broadcasted(::typeof(+), x::Variable, y) = _eladd(x, y) # 新增
# 求局部导数
function ∇(f::ElAdd, gy)  
    gy, gy    
end

# Square
# 创建
@createfunc ElMul
# 求值
_elmul(x, y) = ElMul()(x, y) do x, y
    x .* y
end
# 为已有函数创建新方法
Base.Broadcast.broadcasted(::typeof(*), x::Variable, y::Variable) = _elmul(x, y) 
Base.Broadcast.broadcasted(::typeof(*), x::Variable, y) = _elmul(x, y) # 新增
Base.Broadcast.broadcasted(::typeof(*), x, y::Variable) = _elmul(x, y) # 新增
# 求局部导数
function ∇(f::ElMul, gy)
    x1, x2 = f.inputs
    gy .* x2.value, gy .* x1.value
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
x = Variable(2.0)
y = x .+ [3.0]
println(y)

y =@. 3.0 * x + 1.0
println(y)