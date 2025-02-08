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
  lastlayers = Chain(AdaptiveMeanPool((1, 1)),MLUtils.flatten,Dense(512 => 50),Dense(50 => 6))
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

function getsimplemodel(;previously_loaded = false)
  modelstructure = Chain(
                          Conv((3,3),1 => 4;stride = 3),
                          Conv((5,5),4 => 16;stride = 4,pad = SamePad()),
                          Conv((7,7),16 => 64;stride = 4,pad = SamePad()),
                          AdaptiveMeanPool((1, 1)),MLUtils.flatten,
                          Dense(64 => 20),
                          Dense(20 => 6)
                          )
  if previously_loaded == false
    #zerotrainingepochsbybatchcsv()
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
  model = getsimplemodel(;previously_loaded = !resetmodel)#TODO
  trainingdatainput, trainingdataoutput = gettrainingdatainbatches(; set = "train")
  verificationdatainput, verificationdataoutput = gettrainingdatainbatches(; set = "verify")
  lossfunc = getlossfunc()
  #includes regularization #Flux.setup(Adam(), model)
  return model, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, lossfunc
end

function savemodelstate(model)
  modelstatenumber = findnewestmodelstatenumber() + 1
  filename = lpad(modelstatenumber,6,"0")*".jld2"
  model_state = Flux.state(cpu(model))
  jldsave("PROCESSING/camera/model_states/"*filename; model_state)
  return modelstatenumber
end

function totaldatasetloss(model, lossfunc, trainingdatainput, trainingdataoutput)
  total = Float32(0.0)#TODO: check if model is on GPU
  for batchnumber = 1:1:size(trainingdatainput)[4]
    currentbatchtrainingdataoutput = cu(trainingdataoutput[:,:,batchnumber])
    currentbatchtrainingdatainput = cu(trainingdatainput[:,:,batchnumber:batchnumber,:])
    #for itemnumber = 1:1:size(trainingdatainput)[3]
    total = total + lossfunc(model(currentbatchtrainingdatainput), currentbatchtrainingdataoutput)
    #end
    #CUDA.unsafe_free!(currentbatchtrainingdataoutput)
    #CUDA.unsafe_free!(currentbatchtrainingdatainput)
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

function trainbatch(currentbatchtrainingdatainput,currentbatchtrainingdataoutput,lossfunc,model,opt_state)
  #for i = 1:1:size(currentbatchtrainingdatainput)[3]
    ##=
    grads = Flux.gradient(model) do m
      #result = m(currentbatchtrainingdatainput[:,:,i:i,:])
      lossfunc(m(currentbatchtrainingdatainput[:,:,:,:]), currentbatchtrainingdataoutput[:,:])#x,y,1(was batchnumber),itemnumber     out,itemnumber,  wasbatchnumber
    end
    Flux.update!(opt_state, model, grads[1])
    # =#
 # end
end

function batchgrads(currentbatchtrainingdatainput,currentbatchtrainingdataoutput,lossfunc,model)
  grads = Flux.gradient(model) do m
    #result = m(currentbatchtrainingdatainput[:,:,i:i,:])
    lossfunc(m(currentbatchtrainingdatainput[:,:,:,:]), currentbatchtrainingdataoutput[:,:])#x,y,1(was batchnumber),itemnumber     out,itemnumber,  wasbatchnumber
  end
  return grads
end

function trainmodel(nseries,nepochs;resetmodel = false,epochchunksize = 1)
  CUDA.reclaim()
  reset = deepcopy(resetmodel)
  model, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, lossfunc = getmodelreadytotrain(resetmodel)
  nbatches = size(trainingdatainput)[3]
  nbatchestrainedperseries = nbatches * nepochs
  #batchnumbers, epochseach = getnumberepochsperbatch(nbatchestrainedperseries,epochchunksize)
  c_model = cu(model)
  c_lossfunc = cu(lossfunc) #                       1e-4            1.0        0.001
  opt_state = Flux.setup(OptimiserChain(WeightDecay(1e-4), ClipNorm(1.0), Adam(0.001)), c_model)
  println("Starting training")
  totalepochcounter = 0
  for i = 1:1:nseries
    CUDA.reclaim()
    epochcounter = 0
    for j = 1:1:nepochs
      accum_grads = nothing #TODO: may be a better way of initializing this that doesn't necesitate if statement
      re = nothing
      for batchnumber = 1:1:nbatches
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
      Flux.update!(opt_state, c_model, re(accum_grads)[1])
      epochcounter = epochcounter + 1
      totalepochcounter = totalepochcounter + 1
      println("Completed $epochcounter / $nepochs epochs in series $i / $nseries. ( $totalepochcounter / $(nseries*nepochs) )")
    end
    modelstatenumber = savemodelstate(c_model)
    updatelossbyepochcsv(Int32(floor(nbatchestrainedperseries/nbatches)), c_model, trainingdatainput, trainingdataoutput, verificationdatainput, verificationdataoutput, c_lossfunc, modelstatenumber)
    #updatetrainingepochsbybatchcsv(batchnumbers, epochseach)
    #model = cpu(c_model)
    #c_model = nothing
    #opt_state = nothing
    GC.gc()
    #CUDA.reclaim()
    reset = false
  end
end