# step47

using Judo
using Random
#Random.seed!(0)

model = MLP((10, 3))

x = [0.2 -0.4;0.3 0.5;1.3 -3.2;2.1 0.3]
t = [3,1,2,1]
y = model(x)
loss = softmax_cross_entropy(y,t)

@inference gradient!(loss)
println(loss)
