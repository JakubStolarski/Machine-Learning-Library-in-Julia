# Layers
mutable struct DenseLayer
    in_channels::Int
    out_channels::Int
    weights::Array{Float64}
    activation::Function
    make_flat::Bool
end

mutable struct ConvLayer
    in_channels::Int
    out_channels::Int
    weights::Array{Float64}
    kernel_size::Int
    activation::Function
end

function conv_layer(in_channels::Int, out_channels::Int, kernel_size::Int, activation::Function=nothing)
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float64, kernel_size, kernel_size, in_channels, out_channels)
    return ConvLayer(in_channels::Int, out_channels::Int, weights, kernel_size, activation)
end

function dense_layer(in_channels::Int, out_channels::Int, activation::Function=nothing, make_flat::Bool=false) 
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float32, in_channels, out_channels)
    return DenseLayer(in_channels::Int, out_channels::Int, weights, activation, make_flat)
end

## Functions
using LinearAlgebra
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
    output = weight .* input 
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

        
function softmax(x::Union{Matrix, Vector, Array, Int, LinearAlgebra.Transpose{Float64, Vector{Float64}}}) 
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end
        
        
function int_to_array(x::Int, length::Int)
    output = zeros(length)
    output[x+1] = 1
    return output
end
        
# Loss function        
function cross_entropy_loss(pred::Union{Matrix, Vector, Array}, dest::Union{Int,Variable,Constant})
    num_of_classes = Constant(size(pred)[2])
    label = Constant(int_to_array(dest, num_of_classes))
    loss = -sum(dest .* log.(pred))/num_of_classes
    return loss
end

function cross_entropy_loss(pred::Union{Matrix, Vector, Array}, dest::Vector{Number})
    num_of_classes = size(pred)[2]
    loss = []
    for id in 1:size(dest)
        label = int_to_array(dest[id], num_of_classes)
        push!(loss, -sum(label .* log.(pred[id]))/num_of_classes)
    end
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
    fc1 = dense_layer(16*5*5, 120, relu, true)
    fc2 = dense_layer(120, 84, relu)
    fc3 = dense_layer(84, 10, relu)
    sequence = [conv, conv, Iterators.flatten, fully_connect, fully_connect, fully_connect]
    return ConvNet(conv1, conv2, fc1, fc2, fc3, sequence)
end

using MLDatasets, Images, ImageMagick, Shuffle
    
function load_train_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.traindata(Float64)
    resized_images = []
    for i in 1:size(images)[3]
        img = images[:,:,i]
        resized_img = imresize(img, (32, 32))
        push!(resized_images, resized_img)
    end
    
    num_samples = length(resized_images)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [resized_images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end

    return batches
end
    
    
function load_test_data(shape::Tuple=(32,32), batch_size::Int=64)
    images, labels = MNIST.testdata(Float64)
    resized_images = []
    for i in 1:size(images)[3]
        img = images[:,:,i]
        resized_img = imresize(img, (32, 32))
        push!(resized_images, resized_img)
    end
    
    num_samples = length(images_resized)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [resized_images[j] for j in batch_indices]
        batch_labels = [labels[k] for k in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end
    
    return batches
end


function infer(model::ConvNet, input::Union{Matrix,Vector})
    output =[]
    push!(output,input)
    for layer in fieldnames(typeof(model))
        if isa(getfield(net,layer), ConvLayer)
            field = getfield(net,layer)
            println(size(field.weights))
            new_output = []
                for kernel_set in 1:field.out_channels
                    step_output = []
                    for in_image in 1:field.in_channels
                        if step_output == []
                        step_output = pooling(field.activation(conv(output[in_image],field.weights[:,:,in_image,kernel_set])),2)
                        else
                        step_output .+ pooling(field.activation(conv(output[in_image],field.weights[:,:,in_image,kernel_set])),2)
                        end
                    end
                    push!(new_output,step_output)
                end


            output = new_output
        end
                
        if isa(getfield(model,layer), DenseLayer)
            field = getfield(model,layer)
            if field.make_flat == true
                output = transpose(collect(Iterators.flatten((output))))
            end
            output = output * field.weights  # todo MAKE FC FUNCTION DO THAT, DON"T FORGET ABOUT ACTIVATION FUNCTION
        end
    end
    return output
end
                             
    
function training(num_of_epochs::Int, model::ConvNet, training_data, training_labels, criterion::Function, optimizer::Function)
    for epoch in 1:num_of_epochs
        for training_batch in training_data
            image = training_batch[1][1]
            label = training_batch[1][2]
            
            outputs = infer(model,image)
            loss = criterion(outputs, label)
        end
    end
end


      


net = create_net()
learning_rate = 0.016
num_of_epochs = 10
        
train_loader = load_train_data()
u = train_loader[1][1][1]
criterion = cross_entropy_loss
pred = infer(net,u)
pred = softmax(pred)
d = train_loader[1][2][1]
cross_entropy_loss(pred,d)


# Dummy function
function infer(model::ConvNet, input::Union{Matrix,Vector})
    final_output = []
    for i in 1:size(input)[1]
        image_output = input[i]
        for layer in fieldnames(typeof(model))
            if isa(getfield(model,layer), ConvLayer)
                field = getfield(model,layer)
                new_image_output = []
                for kernel_set in 1:field.out_channels
                    step_output = []
                    for in_image in 1:field.in_channels
                        if step_output == []
                        step_output = pooling(field.activation(conv(image_output[in_image],field.weights[:,:,in_image,kernel_set])),2)
                        else
                        step_output .+ pooling(field.activation(conv(image_output[in_image],field.weights[:,:,in_image,kernel_set])),2)
                    end
                    push!(new_output,step_output)
                end

                image_output = new_output
            end

            if isa(getfield(model,layer), DenseLayer)
                field = getfield(model,layer)
                if field.make_flat == true
                    image_output = collect(Iterators.flatten((output)))
                end
                new_output = []
                for out_neuron in 1:field.out_channels
                    push!(new_output, fully_connect(image_output, field.weights[:,out_neuron], field.activation))      
                end

                image_output = new_output
            end
        end
        ret
end

using Plots

"""
    display_image(image::Array{Float64,2})
Display a 2D array as an image.

# Arguments
- `image::Array{Float64,2}`: The 2D array representing the image.

"""
function display_image(image::Array{Float64,2})
    plot(
        heatmap(image,
                aspect_ratio=:equal,
                frame=:none,
                c=:grays,
                colorbar=false,
                legend=false),
        ticks=nothing,
        border=:none,
        axis=:off
    )
end


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

# Base.Broadcast.broadcasted(maximum, x::GraphNode) = BroadcatedOperator(maximum,x)
# forward(::BroadcastedOperator{typeof(maximum)}, x) = return maximum(x)
# backward(::BroadcastedOperator{typeof(maximum)},x, g) = 

x = Variable(randn(1,400),name="x")
y = Variable(randn(1,10),name="y")
w1 = Variable(net.conv1.weights, name="w1")
w2 = Variable(net.conv2.weights, name="w2")
w3 = Variable(net.fc1.weights, name="w3")
w4 = Variable(net.fc2.weights, name="w4")
w5 = Variable(net.fc3.weights, name="w5")
function testing_net()
    x1 = relu(x*w3)
    x2 = relu(x1*w4)
    x3 = x2*w5
    x4 = softmax(x3)
    x5 = cross_entropy_loss(x4,y)
    return topological_sort(loss)
end
graph = testing_net()



