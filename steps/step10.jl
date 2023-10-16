# step10

using Test

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
_square(x) = Square()(x) do t  
    t.^2                        
end                            

# 为已有函数创建新方法
Base.Broadcast.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::Variable, ::Val{2}) = _square(x)

# 求导
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


function numerical_diff(f, x::Variable, eps=1e-4)
    x0 = Variable(x.value .- eps)
    x1 = Variable(x.value .+ eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.value .- y0.value) ./ 2eps 
end

# test
function allclose(a, b; rtol=1e-5, atol=1e-8)
    return all(@. abs(a - b) <= (atol + rtol * abs(b)))
end

@testset "SquareTest and ExpTest" begin
    # Square
    @testset "evaluation" begin
        x = Variable([2.0])
        y = x.^2
        @test y.value == [4.0]
    end
    @testset "gradient" begin
        function test_grad_check()
            x = Variable(rand(1))
            y = x.^2
            gradient!(y)
            num_grad = numerical_diff(x->x.^2, x)
            @test allclose(x.grad, num_grad)
        end
        for i in 1:10
            test_grad_check()
        end
    end
    # Exp
    @testset "evaluation" begin
        x = Variable([2.0])
        y = exp(x)
        @test y.value ≈ exp.(x.value)
    end
    @testset "gradient" begin
        function test_grad_check()
            x = Variable(rand(1))
            y = exp(x)
            gradient!(y)
            num_grad = numerical_diff(exp, x)
            @test allclose(x.grad, num_grad)
        end
        for i in 1:10
            test_grad_check()
        end
    end
end;
