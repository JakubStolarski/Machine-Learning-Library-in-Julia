using Random

# Define sigmoid function
sigmoid(x) = 1 / (1 + exp(-x))

# Define model architecture
struct Model
    w1::Matrix{Float64}
    b1::Vector{Float64}
    w2::Matrix{Float64}
    b2::Vector{Float64}
end

# Define forward pass through the model
function forward(model::Model, x::Vector{Float64})
    a1 = model.w1 * x .+ model.b1
    h1 = sigmoid.(a1)
    a2 = model.w2 * h1 .+ model.b2
    y = softmax(a2)
    return a1, h1, a2, y
end

# Define cross-entropy loss function
function cross_entropy_loss(y_pred, y_true)
    -sum(y_true .* log.(y_pred))
end

# Define function to calculate gradients with automatic differentiation using a graph
function backward(model::Model, x::Vector{Float64}, y_true::Vector{Float64})
    # Forward pass
    a1, h1, a2, y_pred = forward(model, x)

    # Compute loss
    loss = cross_entropy_loss(y_pred, y_true)

    # Initialize gradients
    grad_w1 = zeros(size(model.w1))
    grad_b1 = zeros(size(model.b1))
    grad_w2 = zeros(size(model.w2))
    grad_b2 = zeros(size(model.b2))

    # Backward pass
    graph = [a1, h1, a2, y_pred]
    gradients = [nothing, nothing, nothing, y_pred - y_true]

    # Backpropagate through each layer
    for i = 3:-1:1
        grad = gradients[i + 1]
        input = graph[i]

        # Compute gradients of weights and biases
        grad_w = grad * transpose(graph[i - 1])
        grad_b = grad

        # Update gradients
        gradients[i] = transpose(model.w[i]) * grad
        if i == 3
            gradients[i] .= gradients[i] .* softmax_gradient(a2)
        else
            gradients[i] .= gradients[i] .* sigmoid_gradient(input)
        end
        grad_w .= grad_w / size(x, 2)
        grad_b .= mean(grad_b, dims=2)

        # Update model parameters
        model.w[i] -= learning_rate * grad_w
        model.b[i] -= learning_rate * grad_b
    end

    return loss, model
end

# Define sigmoid gradient function
function sigmoid_gradient(x)
    s = sigmoid.(x)
    s .* (1 .- s)
end

# Define softmax function
function softmax(x)
    exp.(x) ./ sum(exp.(x))
end

# Define softmax gradient function
function softmax_gradient(x)
    s = softmax(x)
    s .* (1 .- s)
end

# Generate random training data
Random.seed!(123)
x_train = rand(2, 1000) .* 2 .- 1
y_train = (sum(x_train, dims=1) .> 0) .+ 1

# Initialize model
learning_rate = 0.1
model = Model(randn(4, 2), randn(4), randn(2, 4), randn(2))


using LinearAlgebra

function convolution(input::Matrix{Float64}, kernel, activation=nothing)
    input_height, input_width = size(input)
    kernel_height, kernel_width, in_channels, out_channels = size(kernel)
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(output_height, output_width, out_channels)
    #print(size(input))
    #print(size(output))
    for in_channel in in_channels
        for out_channel in out_channels
            for i in 1:output_height
                for j in 1:output_width
                    for k in 1:kernel_height
                        for l in 1:kernel_width
                            output[i, j, out_channel] += input[i+k-1, j+l-1] * kernel[k, l, in_channel, out_channel]
                        end
                    end
                end
            end
        end
    end
    if activation != nothing
        output = activation(output)
    end
     #println(size(output))
    return output
end

function convolution(input::Array{Float64}, kernel, activation=nothing)
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, in_channels, out_channels = size(kernel)
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(output_height, output_width, out_channels)
    #print(size(input))
    #print(size(output))
    for input_channel in input_channels
        for in_channel in in_channels
            for out_channel in out_channels
                for i in 1:output_height
                    for j in 1:output_width
                        for k in 1:kernel_height
                            for l in 1:kernel_width
                                output[i, j, out_channel] += input[i+k-1, j+l-1, input_channel] * kernel[k, l, in_channel, out_channel]
                            end
                        end
                    end
                end
            end
        end
    end
    if activation != nothing
        output = activation(output)
    end
     #println(size(output))
    return output
end

function fully_connected(input, weight, activation=nothing)
    output = weight * input 
    if activation != nothing
        output = activation(output)
    end
    
    return output
end

function relu(x::Union{Matrix, Vector, Array, Int, Transpose{Float64, Vector{Float64}}})
    return max.(0, x)
end

function softmax(x::Union{Matrix, Vector, Array, Int, LinearAlgebra.Transpose{Float64, Vector{Float64}}}) 
    exp_x = exp.(x)
    return exp_x ./ sum(exp_x)
end

function softmax_backwards(grad, output)
    softmax_grad = zeros(size(output))
    for i in 1:size(output, 1)
        for j in 1:size(output, 2)
            if i == j
                softmax_grad[i, j] = output[i, j] * (1 - output[i, j])
            else
                softmax_grad[i, j] = -output[i, j] * output[j, i]
            end
        end
    end
    
    grad_input = softmax_grad * grad
    
    return grad_input
end

function pooling(x::Union{Matrix, Vector, Array, Int}, kernel::Int=2, stride::Int=kernel, pool_type::String="max")
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape, channels))
    #println(size(output))
    for channel in 1:channels
        for i in 1:output_shape
            for j in 1:output_shape
                if pool_type == "max"
                    output[i, j, channel] = maximum(x[i:i+stride,j:j+stride, channel])
                elseif pool_type == "mean"
                    output[i, j, channel] = mean(x[i:i+stride,j:j+stride, channel])
                end
            end
        end
    end
    #println(size(output))        
    return output
end

function cross_entropy_loss(pred::Union{Matrix, Vector, Array}, dest::Int)
    num_of_classes = size(pred)[2]
    label = int_to_array(dest, num_of_classes)
    loss = -sum((label .* log.(pred1)))/num_of_classes
    return loss
end

function cross_entropy_loss_backwards(grad, pred, dest)
    num_of_classes = size(pred)[2]
    label = int_to_array(dest, num_of_classes)
    return (pred - label) * grad
end

mutable struct DenseLayer
    in_channels::Int64
    out_channels::Int64
    weights::Array
    activation::Function
    make_flat::Bool
end

function DenseLayer(in_channels::Int64, out_channels::Int64, activation::Function=nothing, make_flat::Bool=false)
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float32, out_channels, in_channels)
    return DenseLayer(in_channels, out_channels, weights, activation, make_flat)
end

mutable struct ConvLayer
    in_channels::Int64
    out_channels::Int64
    weights::Array
    kernel_size::Int64
    activation::Function
end

function ConvLayer(in_channels::Int64, out_channels::Int64, kernel_size::Int64, activation::Function=nothing)
    weights = -0.2 .+ (0.2 - (-0.2)) .* randn(Float64, kernel_size, kernel_size, in_channels, out_channels)
    return ConvLayer(in_channels, out_channels, weights, kernel_size, activation)
end

struct LeNet5
    conv1::ConvLayer
    conv2::ConvLayer
    fc1::DenseLayer
    fc2::DenseLayer
    fc3::DenseLayer
end

function intialize_network()
    conv1 = ConvLayer(1, 6, 5, relu)
    conv2 = ConvLayer(6, 16, 5, relu)
    fc1 = DenseLayer(16*5*5, 120, relu, true)
    fc2 = DenseLayer(120, 84, relu)
    fc3 = DenseLayer(84, 10, softmax)  
    # Unlike in reference_solution.py, here the softmax function is done at the end of third dense layer instead of 
    # at the beginning of cross entropy loss calculation, operation wise it should make no difference 
    return LeNet5(conv1, conv2, fc1, fc2, fc3)
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

train_data = load_train_data()
u = train_data[1][1][1]
net = intialize_network()
o = forward(net, u)


function forward(net::LeNet5, input::Union{Matrix{Float64},Vector})
    forward_results = []
    x = convolution(input, net.conv1.weights); push!(forward_results,x)
    x = net.conv1.activation(x); push!(forward_results,x)    
    x = pooling(x); push!(forward_results,x)
    x = convolution(x, net.conv2.weights); push!(forward_results,x)
    x = net.conv2.activation(x); push!(forward_results,x)
    x = pooling(x); push!(forward_results,x)
    x = collect(Iterators.flatten(x))
    x = fully_connected(x, net.fc1.weights); push!(forward_results,x)
    x = net.fc1.activation(x); push!(forward_results,x)
    x = fully_connected(x, net.fc2.weights); push!(forward_results,x)
    x = net.fc2.activation(x); push!(forward_results,x)
    x = fully_connected(x, net.fc3.weights); push!(forward_results,x)
    x = net.fc3.activation(x); push!(forward_results,x)
    return forward_results
end


