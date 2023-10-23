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
# 1、创建
@createfunc Exp
# 2、求值+3、扩展
Base.exp(x::Variable) = Exp()(x) do x
    exp.(x)
end
# 4、求导
∇(f::Exp, gy) = gy * f.outputs[1]

# Log
# 1、创建
@createfunc Log
# 2、求值+3、扩展
Base.log(x::Variable) = Log()(x) do x
    log.(x)
end
# 4、求导
∇(f::Log, gy) = gy / f.inputs[1]

# ===================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# ===================================================================

# Reshape
# 1、创建
@createfunc Reshape shape::Tuple
# 2、求值+3、扩展
Base.reshape(x::Variable, shape::Tuple) = size(x) == shape ? x : Reshape(shape)(x) do x
    reshape(x, shape)
end
function Base.reshape(x::Variable, shape...) 
    if length(shape) == 1 && shape[1] isa Union{Tuple,Array}
        shape = shape[1]
    end
    return reshape(x,tuple(shape...))
end
# 4、求导
∇(f::Reshape, gy) = reshape(gy, f.x_shape)

# Transpose
# 1、创建
@createfunc Transpose
# 2、求值+3、扩展
Base.transpose(x::Variable) = Transpose()(x) do x
    transpose(x)
end
# 4、求导
∇(f::Transpose, gy) = transpose(gy)

# Adjoint
# 1、创建
@createfunc Adjoint
# 2、求值+3、扩展
Base.adjoint(x::Variable) = Adjoint()(x) do x
    adjoint(x)
end
# 4、求导
∇(f::Adjoint, gy) = adjoint(gy) # ???


# ===================================================================
# sum / average / matmul / linear /max
# ===================================================================

# Sum
# 1、创建 
@createfunc Sum dims
# 2、求值+3、扩展
Base.sum(x::Variable; dims=:) = Sum(dims)(x) do x
    sum(x, dims=dims)
end
# 4、求导
∇(f::Sum, gy) = broadcastto(gy, f.x_shape)

# MatMul
# 1、创建 
@createfunc MatMul
# 2、求值+3、扩展
matmul(W, x) = MatMul()(W, x) do W, x
    W = length(W) == 1 ? W[1] : W # 当数值只有一个元素时， 取出该值
    x = length(x) == 1 ? x[1] : x # 这样可以防止向量乘以向量的错误
    W * x
end
⋅(A::Variable, B::Variable) = matmul(A, B)
⋅(A::Variable, B) = matmul(A, B)
⋅(A, B::Variable) = matmul(A, B)
# 4、求导
function ∇(f::MatMul, gy)
    W, x = f.inputs
    gW = gy ⋅ x'
    gx = W' ⋅ gy
    return gW, gx
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