# step01

mutable struct Variable # 作为容器的变量
    value               # 和容器中保存的内容 `value`
end

data = 1.0
x = Variable(data)
println(x.value)

x.value = [2.0]
println(x.value)
