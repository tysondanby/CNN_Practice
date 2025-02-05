using Metalhead, Flux, MLUtils, Images, CSV, DataFrames
previously_loaded = false

function lpnormpoolinglayer(x)
    return lpnormpool(x,Float32(0.5),(3,3)) ./ Float32(81)
end

function getfirstlayer()
    return Parallel(function f(a,b,c) return cat(a,b,c;dims=(3)) end,MaxPool((3,3)),MeanPool((3,3)),lpnormpoolinglayer)
end


function getlossfunc()
  return  function lossfunction(yhat,y)
            return sum((yhat .- y).^2)
          end
end

function updatetrainingepochsbybatchcsv(nbatches::Int)
  csvname ="PROCESSING/camera/trainingepochsbybatch.csv"
  historycsv = CSV.File(csvname)
  newbatches = nbatches - length(historycsv.batch)
  if newbatches > 0
    lastbatchnumber = historycsv.batch[end]
    for i = 1:1:newbatches
      push!(historycsv.batch,lastbatchnumber+i)
      push!(historycsv.epochs,0)
    end
    CSV.write(csvname,DataFrame(historycsv))
  end
end

function updatetrainingepochsbybatchcsv(batchestrained,epochstrainedeach)
  csvname ="PROCESSING/camera/trainingepochsbybatch.csv"
  historycsv = CSV.File(csvname)
  for (i, batchnumber) in enumerate(batchestrained)
    historycsv.epochs[batchnumber] = historycsv.epochs[batchnumber] + epochstrainedeach[i]
  end
  CSV.write(csvname,DataFrame(historycsv))
end

function gettrainingdatabatch(batchsize,batchnumber,datacsv)
  batch = []::Vector{Tuple{Array{Float32, 4}, Vector{Float32}}}
  for imagenumber = 1:1:batchsize
    datacsvindex = 1+imagenumber + batchsize*(batchnumber-1)
    imgname = datacsv.filename[datacsvindex]
    img = Float32.(Images.load("DATA/camera/prepared/"*imgname))
    fourdimensionalimage = zeros(Float32,size(img)...,1,1)
    fourdimensionalimage[:,:,1,1] = img
    answer = Float32.([datacsv.x1[datacsvindex],datacsv.y1[datacsvindex],datacsv.x2[datacsvindex],datacsv.y2[datacsvindex],datacsv.x3[datacsvindex],datacsv.y3[datacsvindex]])
    push!(batch,(fourdimensionalimage,answer))
  end
  return batch
end

function gettrainingdatabatches(nbatches,batchsize,datacsv)
  batches = []::Vector{Vector{Tuple{Array{Float32, 4}, Vector{Float32}}}}
  for batchnumber = 1:1:nbatches
    push!(batches,gettrainingdatabatch(batchsize,batchnumber,datacsv))
  end
  return batches
end

function gettrainingdatainbatches(;batchsize = 32, set = "train")
  datacsv = CSV.File("DATA/camera/model_data/"*set*".csv")
  nbatches = floor((length(datacsv.filename)-1)/batchsize)
  updatetrainingepochsbybatchcsv(nbatches)
  return gettrainingdatabatches(nbatches,batchsize,datacsv)
end

function getmodel(;previously_loaded = false)
  if previously_loaded == false
    pretrainmodel = ResNet(18; pretrain = true)
    pretrainedbackbone = pretrainmodel.layers[1]
    firstlayer = getfirstlayer()
    lastlayers = Chain(AdaptiveMeanPool((1, 1)),MLUtils.flatten,Dense(512 => 100),Dense(100 => 6))
    compositemodel = Chain(firstlayer,pretrainedbackbone,lastlayers)
  else
    #TODO: Load from the latest model save-state in PROCESSING/camera/model_states/
  end
  return compositemodel
end

function getnumberepochsperbatch(nepochs,epochchunksize)
  nchunks = floor(nepochs/epochchunksize)
  remainderepochs = nepochs - (epochchunksize*nchunks)
  historycsv = CSV.File("PROCESSING/camera/trainingepochsbybatch.csv")
  batches_old = historycsv.batch
  epochs = copy(historycsv.epochs)

  minindex = sortperm(epochs)[1]
  batches    = [ batches_old[minindex] ]
  epochseach = [ remainderepochs ]
  epochs[minindex] = epochs[minindex] + remainderepochs
  chunk = 1
  while chunk <= nchunks
    minindex = sortperm(epochs)[1]
    push!(batches,batches_old[minindex])
    push!(epochseach,epochchunksize)
    epochs[minindex] = epochs[minindex] + epochchunksize
    chunk = chunk +1
  end
  return batches, epochseach
end

function trainmodel(nepochs;resetmodel = false,epochchunksize = 10)
  model = getmodel(;previously_loaded = !resetmodel)
  trainingdata = gettrainingdatainbatches(; set = "train")#Vector{Vector{Tuple{AbstractMatrix, AbstractVector}}}
  lossfunc = getlossfunc()
  opt_state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model)#includes regularization #Flux.setup(Adam(), model)

  loss_log = []
  batchestrained = Int32[]
  epochstrainedeach = Int32[]
  batches, epochseach = getnumberepochsperbatch(nepochs,epochchunksize)
  for (batchindex, batch) in enumerate(batches)
    push!(batchestrained,batch)
    push!(epochstrainedeach,0)
    currentbatchtrainingdata = trainingdata[batch]
    for epoch in 1:epochseach[batchindex]
      losses = Float32[]
      for (i, data) in enumerate(currentbatchtrainingdata)
        input, correctoutput = data

        val, grads = Flux.withgradient(model) do m
          result = m(input)
          lossfunc(result, correctoutput)
        end
        push!(losses, val)
        if !isfinite(val)
          @warn "loss is $val on item $i" epoch
          continue
        end
        Flux.update!(opt_state, model, grads[1])
      end
      push!(loss_log, sum(losses))
      epochstrainedeach[end] = epochstrainedeach[end] + 1
      if  sum(losses) < 1.0
        println("stopping after $epoch epochs")
        break
      end
    end
  end 
  #TODO: save model state in PROCESSING/camera/model_states/
  #TODO: record results in lossbyepoch.csv
  updatetrainingepochsbybatchcsv(batchestrained,epochstrainedeach)
end

#TODO: where appropriate, freeze some layers of the model
#TODO FIRST:do manual_data_entry until a good amount of training data is availible, then train the model.






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
