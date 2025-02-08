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

function to_cuda(x)
  if x isa Array
      return cu(x)  # Convert CPU arrays to CUDA arrays
  elseif x isa NamedTuple
      return NamedTuple{keys(x)}(to_cuda.(values(x)))  # Recursively process NamedTuple
  elseif x isa Dict
      return Dict(k => to_cuda(v) for (k, v) in x)  # Process Dicts recursively
  else
      return x  # Leave other types unchanged
  end
end

function lossfunction(yhat,y)
  return sum((yhat .- y).^2)
end

function getnewestmodelstate()
  save_state_filenames = readdir("PROCESSING/camera/model_states/")
  save_state_numbers = zeros(Int64,length(save_state_filenames))
  for i = 1:1:length(save_state_numbers)
    save_state_numbers[i] = parse(Int64,save_state_filenames[i][1:end-5])
  end
  return save_state_filenames[findfirst(x->x == maximum(save_state_numbers),save_state_numbers)], maximum(save_state_numbers)
end


function getmodel(;previously_loaded = false)
  pretrainmodel = ResNet(18; pretrain = !previously_loaded)
  pretrainedbackbone = pretrainmodel.layers[1]
  firstlayer = Parallel(function f(a,b,c) return cat(a,b,c;dims=(3)) end,
                        MaxPool((3,3)),
                        MeanPool((3,3)),
                        Conv((3,3), 1 => 1; stride = 3))
  lastlayers = Chain(AdaptiveMeanPool((1, 1)),
                    MLUtils.flatten,
                    Dense(512 => 50),
                    Dense(50 => 6))
  modelstructure = Chain(firstlayer,pretrainedbackbone,lastlayers)
  if previously_loaded == false
    return modelstructure
  else
    newest, ~ = getnewestmodelstate()
    model_state = JLD2.load("PROCESSING/camera/model_states/"*newest, "model_state")
    Flux.loadmodel!(modelstructure, model_state)
    return modelstructure
  end
end

function getsimplemodel(weightdecay,clipnorm,learningrate;previously_loaded = false)
  modelstructure = Chain(
                          Conv((3,3),1 => 4;stride = 3), 
                          Conv((5,5),4 => 16;stride = 4,pad = SamePad()), Dropout(0.1; rng = CUDA.RNG()),
                          Conv((7,7),16 => 64;stride = 4,pad = SamePad()), Dropout(0.05; rng = CUDA.RNG()),
                          AdaptiveMeanPool((1, 1)),MLUtils.flatten,
                          Dense(64 => 20), Dropout(0.02; rng = CUDA.RNG()),
                          Dense(20 => 6)
                          )
  c_modelstructure = cu(modelstructure)
  c_opt_state = Flux.setup(OptimiserChain(WeightDecay(weightdecay), ClipNorm(clipnorm), Adam(learningrate)),c_modelstructure)
  if previously_loaded == false
    return c_modelstructure, c_opt_state
  else
    newest, ~ = getnewestmodelstate()
    model_state = JLD2.load("PROCESSING/camera/model_states/"*newest, "model_state")
    opt_state = JLD2.load("PROCESSING/camera/opt_states/opt_state.jld2","c_opt_state")
    Flux.loadmodel!(modelstructure, model_state)
    #Flux.loadmodel!(c_opt_state, opt_state) #TODO: load in opt state
    return c_modelstructure, c_opt_state
  end
end

function gettrainingdatabatch(batchsize,batchnumber,datacsv)
  inputdatabatch = zeros(Float32,672,672,batchsize)
  outputdatabatch = zeros(Float32,6,batchsize)
  for imagenumber = 1:1:(batchsize/8)
    datacsvindex = Int32(1+imagenumber + (batchsize/8)*(batchnumber-1))
    imgname = datacsv.filename[datacsvindex]
    img = Float32.(Images.load("DATA/camera/prepared/"*imgname))
    fourdimensionalimage = zeros(Float32,size(img)...,1,1)
    fourdimensionalimage[:,:,1,1] = img
    answer = zeros(Float32,6,1)
    answer[:,1] = Float32.([datacsv.x1[datacsvindex],datacsv.y1[datacsvindex],datacsv.x2[datacsvindex],datacsv.y2[datacsvindex],datacsv.x3[datacsvindex],datacsv.y3[datacsvindex]])
    #do 8 transformed copies:
    inputdatabatch[:,:,Int32(imagenumber*8-7)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-7)] = deepcopy(answer)#0deg
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-6)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-6)] = deepcopy(answer)#90deg
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-5)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-5)] = deepcopy(answer)#180deg
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-4)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-4)] = deepcopy(answer)#270deg
    rotate4dimage90!(fourdimensionalimage,answer)
    flip4dimageX!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-3)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-3)] = deepcopy(answer)#0deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-2)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-2)] = deepcopy(answer)#90deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8-1)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8-1)] = deepcopy(answer)#180deg-flipped
    rotate4dimage90!(fourdimensionalimage,answer)
    inputdatabatch[:,:,Int32(imagenumber*8)] = deepcopy(fourdimensionalimage[:,:,1,1])
    outputdatabatch[:,Int32(imagenumber*8)] = deepcopy(answer)#270deg-flipped
  end
  return inputdatabatch, outputdatabatch
end

function gettrainingdatabatches(nbatches,batchsize,datacsv)
  inputdata = zeros(Float32,672,672,nbatches,batchsize)
  outputdata = zeros(Float32,6,batchsize,nbatches)
  for batchnumber = 1:1:nbatches
    inputdata[:,:,batchnumber,:], outputdata[:,:,batchnumber] = gettrainingdatabatch(batchsize,batchnumber,datacsv)
  end
  return inputdata, outputdata
end

function gettrainingdatainbatches(;batchsize = 16, set = "train")#Batchsize MUST be divisible by 8
  datacsv = CSV.File("DATA/camera/model_data/"*set*".csv")
  nbatches = Int32(floor((length(datacsv.filename)-1)*8/batchsize))
  return gettrainingdatabatches(nbatches,batchsize,datacsv)
end

function modelanddatatoRAM(resetmodel,weightdecay,clipnorm,learningrate; batchsize = 32)
  println("Moving objects to memory - Model (0/3)")
  model,opt_state = getsimplemodel(weightdecay,clipnorm,learningrate;previously_loaded = !resetmodel)
  println("Moving objects to memory - Training Data (1/3)")
  trainingdatainput, trainingdataoutput = gettrainingdatainbatches(; set = "train", batchsize = batchsize)
  println("Moving objects to memory - Verification Data (2/3)")
  verificationdatainput, verificationdataoutput = gettrainingdatainbatches(; set = "verify", batchsize = batchsize)
  lossfunc = deepcopy(lossfunction)
  println("Moving objects to memory - Done! (3/3)")
  return model,opt_state,trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, lossfunc
end

function savemodelstate(model)
  ~,newestmodelstatenumber = getnewestmodelstate()
  modelstatenumber = newestmodelstatenumber + 1
  filename = lpad(modelstatenumber,6,"0")*".jld2"
  model_state = Flux.state(cpu(model))
  jldsave("PROCESSING/camera/model_states/"*filename; model_state)
  return modelstatenumber
end

function totaldatasetloss(model, lossfunc, trainingdatainput, trainingdataoutput)
  total = Float32(0.0)
  for batchnumber = 1:1:size(trainingdatainput)[3]
    currentbatchtrainingdataoutput = cu(trainingdataoutput[:,:,batchnumber])
    currentbatchtrainingdatainput = cu(trainingdatainput[:,:,batchnumber:batchnumber,:])
    total = total + lossfunc(model(currentbatchtrainingdatainput), currentbatchtrainingdataoutput)
  end
  return total / (size(trainingdataoutput)[2]*size(trainingdataoutput)[3])
end

function updatelossbyepochcsv(nepochs, model, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, lossfunc, modelstatenumber)
  trainingloss = totaldatasetloss(model, lossfunc, trainingdatainput, trainingdataoutput)
  verificationloss = totaldatasetloss(model, lossfunc, verificationdatainput, verificationdataoutput)
  csv = CSV.File("PROCESSING/camera/lossbyepoch.csv")
  lastepoch = csv.epoch[end]
  push!(csv.epoch, lastepoch + nepochs)
  push!(csv.trainloss,trainingloss)
  push!(csv.verifyloss,verificationloss)
  push!(csv.modelstatenumber,modelstatenumber)
  CSV.write("PROCESSING/camera/lossbyepoch.csv",DataFrame(csv))
end

function batchgrads(currentbatchtrainingdatainput,currentbatchtrainingdataoutput,lossfunc,model)
  grads = Flux.gradient(model) do m
    lossfunc(m(currentbatchtrainingdatainput[:,:,:,:]), currentbatchtrainingdataoutput[:,:])#x,y,1(was batchnumber),itemnumber     out,itemnumber,  wasbatchnumber
  end
  return grads
end

function trainepoch!(c_opt_state, c_model,trainingdatainput,trainingdataoutput,c_lossfunc)
  accum_grads = nothing
  re = nothing
  for batchnumber = 1:1:size(trainingdatainput)[3]
    currentbatchtrainingdataoutput = cu(trainingdataoutput[:,:,batchnumber])
    currentbatchtrainingdatainput = cu(trainingdatainput[:,:,batchnumber:batchnumber,:])
    grads = batchgrads(currentbatchtrainingdatainput,currentbatchtrainingdataoutput,c_lossfunc,c_model)
    flat_grads, re = Flux.destructure(grads)
    if accum_grads == nothing
      accum_grads = flat_grads
    else
      accum_grads .+= flat_grads
    end
  end
  Flux.update!(c_opt_state, c_model, re(accum_grads)[1])
end

function cleanVRAM()
  println("\n \n"*"Freeing unused GPU rescources")
  GC.gc()
  CUDA.reclaim()
end

function loadoptstate(resetmodel, c_model, optimizer)
  c_opt_state = nothing
  if resetmodel == true
    return Flux.setup(optimizer, c_model)
  else
    @load "PROCESSING/camera/opt_states/opt_state.jld2" c_opt_state
    return to_cuda(c_opt_state)
  end
end

function trainmodel(nseries,nepochs;resetmodel = false,batchsize = 16, weightdecay = 1e-4, clipnorm = 1.0, learningrate = 0.005)
  CUDA.reclaim()
  c_model, c_opt_state, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, lossfunc = modelanddatatoRAM(resetmodel,weightdecay,clipnorm,learningrate;batchsize = batchsize)
  println("Moving model to GPU - (0/1)")
  #c_model = cu(model)
  c_lossfunc = cu(lossfunc)

  # = loadoptstate(resetmodel, c_model, OptimiserChain(WeightDecay(weightdecay), ClipNorm(clipnorm), Adam(learningrate)))
  println("Moving model to GPU - Done! (1/1)")
  totalepochcounter = 0
  for i = 1:1:nseries
    cleanVRAM()
    epochcounter = 0
    println("Initiating Training")
    for j = 1:1:nepochs
      trainepoch!(c_opt_state, c_model,trainingdatainput,trainingdataoutput,c_lossfunc)
      epochcounter = epochcounter + 1
      totalepochcounter = totalepochcounter + 1
      println("Completed $epochcounter / $nepochs epochs in series $i / $nseries. ( $totalepochcounter / $(nseries*nepochs) )")
    end
    println("Saving model checkpoint.")
    modelstatenumber = savemodelstate(c_model)
    @save "PROCESSING/camera/opt_states/opt_state.jld2" c_opt_state
    println("Model checkpoint saved. Calculating and recording model losses.")
    updatelossbyepochcsv(nepochs, c_model, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, c_lossfunc, modelstatenumber)
  end
end