# ==================================================================
# Basic functions: sin / cos / tanh / exp / log
# =====================================================================

# Sin
# 1、创建
@createfunc Sin
# 2、求值+3、扩展
Base.sin(x::Variable) = Sin()(x) do x
    sin.(x)
end
# 4、求导
∇(f::Sin, gy) = gy * cos(f.inputs[1])

# Cos
# 1、创建
@createfunc Cos
# 2、求值+3、扩展
Base.cos(x::Variable) = Cos()(x) do x
    cos.(x)
end
# 4、求导
∇(f::Cos, gy) = gy * -sin(f.inputs[1])

# Tanh
# 1、创建
@createfunc Tanh
# 2、求值+3、扩展
Base.tanh(x::Variable) = Tanh()(x) do x
    tanh.(x)
end
# 4、求导
∇(f::Tanh, gy) = gy * (1 - f.outputs[1]^2)

# Exp
@createfunc Exp
Base.exp(x::Variable) = Exp()(x) do x
    exp.(x.data)
end
backward(f::Exp, gy) = gy .* f.outputs[1]

# Log
@createfunc Log
Base.log(x::Variable) = Log()(x) do x
    log.(x.data)
end
backward(f::Log, gy) = gy ./ f.inputs[1]

# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================

# Reshape
@createfunc Reshape shape::Tuple
Base.reshape(x::Variable, shape::Tuple) = size(x) == shape ? x : Reshape(shape)(x) do x
    reshape(x.data, shape)
end
function Base.reshape(x::Variable, shape...) 
    if length(shape) == 1 && shape[1] isa Union{Tuple,Array}
        shape = shape[1]
    end
    return reshape(x,tuple(shape...))
end
backward(f::Reshape, gy) = reshape(gy, f.x_shape)

# Transpose
@createfunc Transpose
Base.transpose(x::Variable) = Transpose()(x) do x
    transpose(x.data)
end
backward(f::Transpose, gy) = transpose(gy)

# Adjoint
@createfunc Adjoint
Base.adjoint(x::Variable) = Adjoint()(x) do x
    adjoint(x.data)
end
backward(f::Adjoint, gy) = adjoint(gy) # ???


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear /max
# =============================================================================

# Sum
@createfunc Sum dims::Union{Int,Tuple,Nothing}
backward(f::Sum, gy) = broadcastto(gy, f.x_shape)
Base.sum(x::Variable; dims=nothing) = Sum(dims)(x) do x
    if dims isa Nothing
        y = sum(x.data)
    else
        y = sum(x.data, dims=dims)
    end
    return y
end
    
# Max
@createfunc Max
backward(f::Max, gy) = begin #该函数的实现是不合理的，后面需要改进
    x1, x2 = f.inputs
    y = f.outputs[1]
    ET = eltype(gy)
    T = typeof(gy)
    gx1 = T(ET.(x1.data .== y.data))
    gx2 = T(ET.(x2.data .== y.data))
    if f.x_shape[1] != f.x_shape[2]    # 当维数不一样时，需要累加运算
        gx1 = sumto(gx1, f.x_shape[1])
        gx2 = sumto(gx2, f.x_shape[2])
    end

    return gx1, gx2
end
varmax(x1, x2) = Max()(x1, x2) do x1, x2
    max.(x1.data, x2.data)
end
Base.max(x::Variable, y::Variable) = varmax(x, y)
Base.max(x::Variable, y) = varmax(x, y)
Base.max(x, y::Variable) = varmax(x, y)