#-----ENVIRONMENT SETUP-------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__) 
required_packages = ["Metalhead", "Flux", "MLUtils", "Images", "CSV", "DataFrames", "CUDA", "cuDNN", "JLD2"]
function is_installed(pkg)
  return haskey(Pkg.project().dependencies, pkg)
end
function ensure_packages(packages)
    for pkg in packages
        if !is_installed(pkg)
            println("Installing missing package: ", pkg)
            Pkg.add(pkg)
        end
    end
end
ensure_packages(required_packages)
Pkg.instantiate() 
working_dir = dirname(dirname(@__DIR__))
cd(working_dir)
#-----END ENVIRONMENT SETUP---------------------------------------------------

using Metalhead, Flux, MLUtils, Images, CSV, DataFrames, CUDA, cuDNN, JLD2
CUDA.allowscalar(false)
include(working_dir*"/LIB/imageprocessing.jl")
previously_loaded = false

function lpnormpoolinglayer(x)#causes issues with GPU
    return lpnormpool(x,Float32(0.5),(3,3)) ./ Float32(81)
end

function getfirstlayer()
    return Parallel(function f(a,b,c) return cat(a,b,c;dims=(3)) end,MaxPool((3,3)),MeanPool((3,3)),Conv((3,3), 1 => 1; stride = 3))
end


function getlossfunc()
  return  function lossfunction(yhat,y)
            return sum((yhat .- y).^2)
          end
end

function updatetrainingepochsbybatchcsv(nbatches::T) where T<: Integer
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

function zerotrainingepochsbybatchcsv()
  csvname ="PROCESSING/camera/trainingepochsbybatch.csv"
  historycsv = CSV.File(csvname)
  for i = 1:1:length(historycsv.epochs)
    historycsv.epochs[i] = 0
  end
  CSV.write(csvname,DataFrame(historycsv))
end

function gettrainingdatabatch(batchsize,batchnumber,datacsv)
  batch = Tuple{Array{Float32, 4}, Vector{Float32}}[]
  for imagenumber = 1:1:(batchsize/8)
    datacsvindex = Int32(1+imagenumber + (batchsize/8)*(batchnumber-1))
    imgname = datacsv.filename[datacsvindex]
    img = Float32.(Images.load("DATA/camera/prepared/"*imgname))
    fourdimensionalimage = zeros(Float32,size(img)...,1,1)
    fourdimensionalimage[:,:,1,1] = img
    answer = Float32.([datacsv.x1[datacsvindex],datacsv.y1[datacsvindex],datacsv.x2[datacsvindex],datacsv.y2[datacsvindex],datacsv.x3[datacsvindex],datacsv.y3[datacsvindex]])
    #do 8 transformed copies:
    push!(batch,deepcopy((fourdimensionalimage,answer)))#0deg
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#90deg
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#180deg
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#270deg
    rotate4dimage90!(fourdimensionalimage,answer)
    flip4dimageX!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#0deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#90deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#180deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    push!(batch,deepcopy((fourdimensionalimage,answer)))#270deg-flipped
  end
  return batch
end

function gettrainingdatabatches(nbatches,batchsize,datacsv)
  batches = Vector{Tuple{Array{Float32, 4}, Vector{Float32}}}[]
  for batchnumber = 1:1:nbatches
    push!(batches,gettrainingdatabatch(batchsize,batchnumber,datacsv))
  end
  return batches
end

function gettrainingdatainbatches(;batchsize = 32, set = "train")#Batchsize MUST be divisible by 8
  datacsv = CSV.File("DATA/camera/model_data/"*set*".csv")
  nbatches = Int32(floor((length(datacsv.filename)-1)*8/batchsize))
  updatetrainingepochsbybatchcsv(nbatches)
  return gettrainingdatabatches(nbatches,batchsize,datacsv)
end

function findnewestmodelstatefilename()
  save_state_filenames = readdir("PROCESSING/camera/model_states/")
  save_state_numbers = zeros(Int64,length(save_state_filenames))
  for i = 1:1:length(save_state_numbers)
    save_state_numbers[i] = parse(Int64,save_state_filenames[i][1:end-5])
  end
  return save_state_filenames[findfirst(x->x == maximum(save_state_numbers),save_state_numbers)] 
end

function findnewestmodelstatenumber()
  save_state_filenames = readdir("PROCESSING/camera/model_states/")
  save_state_numbers = zeros(Int64,length(save_state_filenames))
  for i = 1:1:length(save_state_numbers)
    save_state_numbers[i] = parse(Int64,save_state_filenames[i][1:end-5])
  end
  return maximum(save_state_numbers)
end

function getmodel(;previously_loaded = false)
  pretrainmodel = ResNet(18; pretrain = !previously_loaded)
  pretrainedbackbone = pretrainmodel.layers[1]
  firstlayer = getfirstlayer()
  lastlayers = Chain(AdaptiveMeanPool((1, 1)),MLUtils.flatten,Dense(512 => 100),Dense(100 => 6))
  modelstructure = Chain(firstlayer,pretrainedbackbone,lastlayers)
  if previously_loaded == false
    zerotrainingepochsbybatchcsv()
    return modelstructure
  else
    newest = findnewestmodelstatefilename()
    model_state = JLD2.load("PROCESSING/camera/model_states/"*newest, "model_state")
    Flux.loadmodel!(modelstructure, model_state)
    return modelstructure
  end
end

function getnumberepochsperbatch(nepochs,epochchunksize)
  nchunks = floor(nepochs/epochchunksize)
  remainderepochs = nepochs - (epochchunksize*nchunks)
  historycsv = CSV.File("PROCESSING/camera/trainingepochsbybatch.csv")
  batches_old = historycsv.batch
  epochs = deepcopy(historycsv.epochs)

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

function getmodelreadytotrain(resetmodel)
  model = cu(getmodel(;previously_loaded = !resetmodel))
  trainingdata = gettrainingdatainbatches(; set = "train")
  verificationdata = gettrainingdatainbatches(; set = "verify")
  lossfunc = cu(getlossfunc())
  opt_state = Flux.setup(OptimiserChain(WeightDecay(0.42), Adam(0.1)), model)#includes regularization #Flux.setup(Adam(), model)
  return model, trainingdata, verificationdata, lossfunc, opt_state
end

function savemodelstate(model)
  modelstatenumber = findnewestmodelstatenumber() + 1
  filename = lpad(modelstatenumber,6,"0")*".jld2"
  model_state = Flux.state(cpu(model))
  jldsave("PROCESSING/camera/model_states/"*filename; model_state)
  return modelstatenumber
end

function totaldatasetloss(model, lossfunc, trainingdata)
  total = Float32(0.0)
  model = cu(model)
  for batchnumber = 1:1:length(trainingdata)
    batch = trainingdata[batchnumber]
    for itemnumber = 1:1:length(batch)
      input, correctoutput = cu(batch[itemnumber])
      total = total + lossfunc(model(input), correctoutput)
    end
  end
  return total
end

function updatelossbyepochcsv(nepochs,model, trainingdata, verificationdata, lossfunc, modelstatenumber)
  trainingloss = totaldatasetloss(model, lossfunc, trainingdata)
  verificationloss = totaldatasetloss(model, lossfunc, verificationdata)
  csv = CSV.File("PROCESSING/camera/lossbyepoch.csv")
  lastepoch = csv.epoch[end]
  push!(csv.epoch, lastepoch + nepochs)
  push!(csv.trainloss,trainingloss)
  push!(csv.verifyloss,verificationloss)
  push!(csv.modelstatenumber,modelstatenumber)
  CSV.write("PROCESSING/camera/lossbyepoch.csv",DataFrame(csv))
end

function trainbatch(currentbatchtrainingdata,lossfunc,model,opt_state)
  losses = zeros(Float32,length(currentbatchtrainingdata))
  for (i, data) in enumerate(currentbatchtrainingdata)
    input, correctoutput = cu(data)
    val, grads = Flux.withgradient(model) do m
      result = m(input)
      lossfunc(result, correctoutput)
    end
    losses[i] = val
    Flux.update!(opt_state, model, grads[1])
  end
  return losses
end

function trainmodel(nepochs;resetmodel = false,epochchunksize = 10, seriesnumber = 1, nseries = 1)
  model, trainingdata, verificationdata, lossfunc, opt_state = getmodelreadytotrain(resetmodel)
  batchnumbers, epochseach = getnumberepochsperbatch(nepochs,epochchunksize)
  println("Starting training")
  epochcounter = 0
  for (batchindex, batchnumber) in enumerate(batchnumbers)
    currentbatchtrainingdata = trainingdata[batchnumber]
    for epoch in 1:epochseach[batchindex]
      losses = trainbatch(currentbatchtrainingdata,lossfunc,model,opt_state)
      epochcounter = epochcounter + 1
      println("Epoch $epochcounter / $nepochs complete in series $seriesnumber / $nseries")
    end
  end
  modelstatenumber = savemodelstate(model)
  updatelossbyepochcsv(nepochs, model, trainingdata, verificationdata, lossfunc, modelstatenumber)
  updatetrainingepochsbybatchcsv(batchnumbers, epochseach)
end

function trainseries(nseries,nepochsperseries;resetmodel = false)
  reset = deepcopy(resetmodel)
  for i = 1:1:nseries
    trainmodel(nepochsperseries;resetmodel = reset,epochchunksize = 10, seriesnumber = i, nseries = nseries)
    reset = false
  end
end

#TODO: consider using the 3rd and fourth dimensions of a fourdimage to indicate the item and batch number. this may make GPU training faster, but several functions must be rewriten
