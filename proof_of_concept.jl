# Layers
mutable struct DenseLayer
    in_channels::Int
    out_channels::Int
    weights::Array{Number}
    activation::Function
end

mutable struct ConvLayer
    in_channels::Int
    out_channels::Int
    weights::Array{Number}
    kernel_size::Int
    activation::Function
end

function conv_layer(in_channels::Int, out_channels::Int, kernel_size::Int, activation::Function=nothing)
    weights = randn(Float32, kernel_size, kernel_size, in_channels, out_channels)
    return ConvLayer(in_channels::Int, out_channels::Int, weights, kernel_size, activation)
end

function dense_layer(in_channels::Int, out_channels::Int, activation::Function=nothing) 
    weights = randn(Float32, in_channels, out_channels)
    return DenseLayer(in_channels::Int, out_channels::Int, weights, activation)
end

## Functions

# Layer functions
function conv(input, kernel, activation=nothing)
    input_height, input_width = size(input)
    kernel_height, kernel_width = size(kernel)
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(output_height, output_width)

    for i in 1:output_height
        for j in 1:output_width
            for k in 1:kernel_height
                for l in 1:kernel_width
                    output[i, j] += input[i+k-1, j+l-1] * kernel[k, l]
                end
            end
        end
    end
    
    if activation != nothing
        output = activation(output)
    end
    
    return output
end

function fully_connect(input, weight, activation=nothing)
    output = weight * input 
    if activation != nothing
        output = activation(output)
    end
    
    return output
end

# Activation functions
function relu(x::Union{Matrix, Vector, Array, Int})
    return max.(0, x)
end

function pooling(x::Union{Matrix, Vector, Array, Int}, kernel::Int=2, stride::Int=kernel, type::String="max")
    a = size(x)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape))

    for i in 1:output_shape
        for j in 1:output_shape
            if type == "max"
                output[i, j] = maximum(x[i:i+stride,j:j+stride])
            elseif type == "mean"
                output[i, j] = mean(x[i:i+stride,j:j+stride])
            end
        end
    end
    
    return output
end

        
function softmax(x::Union{Matrix, Vector, Array, Int}) 
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end
        
        
function int_to_array(x::Int, length::Int)
    output = zeros(length)
    output[x] = 1
    return output
end
        
# Loss function        
function cross_entropy_loss(pred::Union{Matrix, Vector, Array, Int}, dest::Int)
    num_of_classes = size(pred,1)
    label = int_to_array(d, num_of_classes)
    loss = -sum(dest .* log.(pred))/N
    return loss
end
        
# Optimizer  
function SGD!(w::Union{Matrix, Vector, Array, Int}, lr::Number, g::Matrix)
    w .-= lr .* dw
end

# Network
struct ConvNet
    conv1::ConvLayer
    conv2::ConvLayer
    fc1::DenseLayer
    fc2::DenseLayer
    fc3::DenseLayer
    sequence::Array{Function}
end
function create_net()
    conv1 = conv_layer(1, 6, 5, relu)
    conv2 = conv_layer(6, 16, 5, relu)
    fc1 = dense_layer(16*5*5, 120, relu)
    fc2 = dense_layer(120, 84, relu)
    fc3 = dense_layer(84, 10, relu)
    sequence = [conv, conv, fully_connect, fully_connect, fully_connect]
    return ConvNet(conv1, conv2, fc1, fc2, fc3)
end

x = [1.0 2.0 3.0 -4.0;
     5.0 -6.0 7.0 8.0;
     9.0 -10.0 -11.0 12.0;
     -13.0 14.0 15.0 16.0]
kernel = 2
stride = 1
out = pooling(x)

abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

# Graph building

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

# Forward and Backward pass
reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)


function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

# Scalar

import Base: ^, sin, *, sum, max, maximum
import LinearAlgebra: mul!
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

# Broadcasted

# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g,-g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcated(maximum, x::GraphNode) = BroadcatedOperator(maximum,x)
forward(::BroadcastedOperator{typeof(maximum)}, x) = return maximum(x)
backward(::BroadcastedOperator{typeof(maximum)},x, g) = 

using MLDatasets, Images, ImageMagick

function load_train_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.traindata()
    images_resized = [imresize(image, shape) for image in images]
    
    num_samples = length(images_resized)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(2:num_samples)
    
    for id in indices
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end
    
    return batches
end
    
    
function load_test_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.testdata()
    images_resized = [imresize(image, shape) for image in images]
    
    num_samples = length(images_resized)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(2:num_samples)
    
    for id in indices
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end
    
    return batches
end

function infer(model::ConvNet, input::Matix)
    output = []
    for layer in fieldnames(typeof(ConvNet)):
        if typeof(getfield(model,layer)) isa ConvNet
            for i in 1:model.layer.out_channels
                push!(output, pooling(conv(output,model.layer.kernel_size,model.layer.weights),2))
                            
    x = conv2d(x, model.conv1.weight) 
    x = model.conv1.activation.(x)
    x = maxpool(x, (2, 2))
    x = conv2d(x, model.conv2.weight)
    x = model.conv2.activation.(x)
    x = maxpool(x, (2, 2))
    x = reshape(x, :, size(x, 4))
    x = model.fc1.activation.(model.fc1.weight' * x .+ model.fc1.bias)
    x = model.fc2.activation.(model.fc2.weight' * x .+ model.fc2.bias)
    return x
            
    conv1 = conv_layer(1,6,5,relu)
    conv2 = conv_layer(6,16,5,relu)
    fc1 = dense_layer(16*5*5, 120,relu)
    fc2 = dense_layer(120, 84,relu)
    fc3 = dense_layer(84, 10,relu)
            
            
function training(num_of_epochs::Int, model::ConvNet, training_data, training_labels, criterion::Function, optimizer::Function)
    for epoch in 1:num_of_epochs
        for training_batch in training_data
            image = training_batch[1]
            label = training_batch[2]
                    
            out
        
end

net = create_net()
learning_rate = 0.016
num_of_epochs = 10
train_loader = load_train_data()
test_loader = load_test_data()



net = create_net()
net.conv1.activation([-2])

a = randn(Float32, 5, 5, 1, 6)
b = randn(Float32, 32, 32)
c = []
for i in 1:6
    push!(c, pooling(conv(b, a[:,:,1,i],relu),2))
end
d = randn(Float32, 5, 5, 6, 16)
e = []
size(c)
# d[:,:,1,1]
    
for (j,c_layer) in enumerate(c)
    for k in 1:16
        push!(e, pooling(conv(c_layer, d[:,:,j,k],relu),2))
    end
end
(u,v) = enumerate(a)
u




