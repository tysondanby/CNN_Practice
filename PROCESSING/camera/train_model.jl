using Metalhead, Flux
previously_loaded = true

function lpnormpoolinglayer(x)
    return lpnormpool(x,Float32(0.5),(3,3)) ./ Float32(81)
end

function getfirstlayer()
    return Parallel(function f(a,b,c) return cat(a,b,c;dims=(3)) end,MaxPool((3,3)),MeanPool((3,3)),lpnormpoolinglayer)
end

function getdata(;set = "training")
end

if previously_loaded == false
    pretrainmodel = ResNet(18; pretrain = true)
    pretrainedbackbone = pretrainmodel.layers[1]
    firstlayer = getfirstlayer()
    lastlayers = Chain(AdaptiveMeanPool((1, 1)),MLUtils.flatten,Dense(512 => 100),Dense(100 => 6))
    compositemodel = Chain(firstlayer,pretrainedbackbone,lastlayers)
else
end

#=
global model = compositemodel#getmodel()
batchsize = 1
data = [(rand(Float32, 672, 672, 1, batchsize), rand(Float32, 6, 1, 1, batchsize))
        for _ in 1:3]
opt = Optimisers.Adam()
state = Optimisers.setup(opt, model);  # initialise this optimiser's state
for (i, (image, y)) in enumerate(data)
    @info "Starting batch $i ..."
    gs, _ = Flux.gradient(model, image) do m, x  # calculate the gradients
        logitcrossentropy(m(x), y)
    end
    state, global model = Optimisers.update(state, model, gs);
end
=#
##=
model = compositemodel#getmodel()
trainingdata = gettrainingdata(;set = "training")#Vector{Tuple{AbstractMatrix, AbstractVector}}
lossfunc = getlossfunc()
opt_state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model)#includes regularization #Flux.setup(Adam(), model)

my_log = []
for epoch in 1:10
  losses = Float32[]
  for (i, data) in enumerate(trainingdata)
    input, correctoutput = data

    val, grads = Flux.withgradient(model) do m
      # Any code inside here is differentiated.
      # Evaluation of the model and loss must be inside!
      result = m(input)
      lossfunc(result, correctoutput)
    end

    # Save the loss from the forward pass. (Done outside of gradient.)
    push!(losses, val)

    # Detect loss of Inf or NaN. Print a warning, and then skip update!
    if !isfinite(val)
      @warn "loss is $val on item $i" epoch
      continue
    end

    Flux.update!(opt_state, model, grads[1])
  end

  # Compute some accuracy, and save details as a NamedTuple
  #acc = my_accuracy(model, trainingdata)
  #acc,
  push!(my_log, (; losses))

  # Stop training when loss below 1
  if  losses[end] < 1.0
    println("stopping after $epoch epochs")
    break
  end
end # =#












#=
ResNet(
  Chain(
    Chain([
      Conv((7, 7), 3 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
      BatchNorm(64, relu),              # 128 parameters, plus 128
      MaxPool((3, 3), pad=1, stride=2),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),          # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),                # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),          # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),                # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
          BatchNorm(128, relu),         # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),               # 256 parameters, plus 256
        ),
        Chain([
          Conv((1, 1), 64 => 128, stride=2, bias=false),  # 8_192 parameters
          BatchNorm(128),               # 256 parameters, plus 256
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128, relu),         # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),               # 256 parameters, plus 256
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
          BatchNorm(256, relu),         # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),               # 512 parameters, plus 512
        ),
        Chain([
          Conv((1, 1), 128 => 256, stride=2, bias=false),  # 32_768 parameters
          BatchNorm(256),               # 512 parameters, plus 512
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256, relu),         # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),               # 512 parameters, plus 512
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
          BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),               # 1_024 parameters, plus 1_024
        ),
        Chain([
          Conv((1, 1), 256 => 512, stride=2, bias=false),  # 131_072 parameters
          BatchNorm(512),               # 1_024 parameters, plus 1_024
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512, relu),         # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),               # 1_024 parameters, plus 1_024
        ),
        identity,
      ),
    ]),
    Chain(
      AdaptiveMeanPool((1, 1)),
      MLUtils.flatten,
      Dense(512 => 1000),               # 513_000 parameters
    ),
  ),
)         # Total: 62 trainable arrays, 11_689_512 parameters,
          # plus 40 non-trainable, 9_600 parameters, summarysize 44.642 MiB.
          =#
