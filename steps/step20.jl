# step20

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
    name       # 增加了名字域
    # 重新定义了内部构造函数, 只接受两个参数
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

# 为新构建的数据类型， 扩展常用函数的方法
Base.ndims(v::Variable) = ndims(v.value)
Base.size(v::Variable) = size(v.value)
Base.size(v::Variable, i) = size(v.value, i)
Base.eltype(v::Variable)  = eltype(v.value)
Base.length(v::Variable) = length(v.value)
Base.show(io::IO, v::Variable) = println(io, "Variable(", v.value, ")")
function Base.show(io::IO, ::MIME"text/plain", v::Variable) 
    println(io, "Variable\n", v.value)
end

abstract type Func end  

# 由于函数类型迅速增加， 每次都手动设置十分耗时， 因此创建一个宏， 用来快速创建函数类型
macro createfunc(name, arg...) # `name` 是函数类型名， `arg` 是共同域以外的独有域
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
    xs = [input.value for input in inputs]   
    ys = fun(xs...)   
    !isa(ys, Tuple) && (ys = tuple(ys))
    outputs = Variable.(ys) 

    if enable_grad[] 
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

# ElAdd
#创建
@createfunc ElAdd
# 求值
_eladd(x, y) = ElAdd()(x, y) do x, y  
    x .+ y
end                            
# 为已有函数创建新方法
Base.Broadcast.broadcasted(::typeof(+), x::Variable, y::Variable) = _eladd(x, y)
# 求局部导数
function ∇(f::ElAdd, gy)  
    gy, gy    
end

# Square
# 创建
@createfunc ElMul
# 求值
_elmul(x1, x2) = ElMul()(x1, x2) do x1, x2
    x1 .* x2
end
# 为已有函数创建新方法
Base.Broadcast.broadcasted(::typeof(*), x1::Variable, x2::Variable) = _elmul(x1, x2) 
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
a = Variable(3.0)
b = Variable(2.0)
c = Variable(1.0)

y =@. a * b + c

gradient!(y)

println(y)
println(a.grad)
println(b.grad)