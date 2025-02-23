include("train_model.jl")
global gpubatchespersuperbatch = 4 #depends on RAM
global gpubatchsize = 16#depends on VRAM
function getinputdatabatch(filenames)
    imagebatch = zeros(Float32,672,672,1,length(filenames))
    i = 1
    for filename in filenames
        #imgname = datacsv.filename[datacsvindex]
        img = Float32.(Images.load("DATA/camera/prepared/"*filename))
        fourdimensionalimage = zeros(Float32,size(img)...,1,1)
        fourdimensionalimage[:,:,1,1] = img
        propnumber = getpropellerfromimagename(filename)
        if (propnumber == 1) || (propnumber == 5)
            flip4dimageX!(fourdimensionalimage)
        end
        imagebatch[:,:,:,i] = fourdimensionalimage
        i = i + 1
    end
    return imagebatch
end
function getinputdatasuperbatch(filenames,nbatches)#;batchespersuperbatch = gpubatchespersuperbatch,batchsize = gpubatchsize)
    batchsize = Int(length(filenames)/nbatches)
    imagesuperbatch = zeros(Float32,672,672,nbatches,batchsize)
    for i = 1:1:nbatches
        batchfilenames = filenames[1+batchsize*(i-1):batchsize*i]
        imagesuperbatch[:,:,i,:] = getinputdatabatch(batchfilenames)
    end
    return imagesuperbatch
end

function evalmodel(imagesuperbatch,c_model)
    resultssuperbatch = zeros(Float32,5,size(imagesuperbatch)[3],size(imagesuperbatch)[4])
    for i = 1:1:(size(imagesuperbatch)[3])
        resultssuperbatch[:,i,:] = cpu(c_model(cu(imagesuperbatch[:,:,i:i,:])))
    end
    return resultssuperbatch
end

function recordresults(filenames,resultssuperbatch,csvfile)
    batchsize = size(resultssuperbatch)[3]
    for batch = 1:1:size(resultssuperbatch)[2]
        for item = 1:1:batchsize
            push!(csvfile.filename,filenames[batchsize*(batch-1)+item])
            push!(csvfile.xc,resultssuperbatch[1,batch,item])
            push!(csvfile.yc,resultssuperbatch[2,batch,item])
            push!(csvfile.r,resultssuperbatch[3,batch,item])
            push!(csvfile.sin,resultssuperbatch[4,batch,item])
            push!(csvfile.cos,resultssuperbatch[5,batch,item])
        end
    end
    CSV.write("DATA/camera/model_data/model_results.csv",DataFrame(csvfile))
end

function processdatawithmodel(;batchespersuperbatch = gpubatchespersuperbatch,batchsize = gpubatchsize)
    superbatchsize = batchespersuperbatch*batchsize
    filenames = readdir("DATA/camera/prepared")
    nsuperbatches = Int(floor(length(filenames)/superbatchsize))
    remainder = length(filenames) -(nsuperbatches*superbatchsize)
    model, ~ = getmodel(0.0,0.0,.001;previously_loaded = true)
    c_model = cu(model)
    for i = 1:1:nsuperbatches
        imagesuperbatch = getinputdatasuperbatch(filenames[1+superbatchsize*(i-1):superbatchsize*i],batchespersuperbatch)
        resultssuperbatch = evalmodel(imagesuperbatch,c_model)
        recordresults(filenames[1+superbatchsize*(i-1):superbatchsize*i],resultssuperbatch,CSV.File("DATA/camera/model_data/model_results.csv"))
    end
    if remainder > 0
        remainderfilenames = filenames[end-(remainder-1):end]
        imagesuperbatch = getinputdatasuperbatch(remainderfilenames,1)
        resultssuperbatch = evalmodel(imagesuperbatch,c_model)
        recordresults(remainderfilenames,resultssuperbatch,CSV.File("DATA/camera/model_data/model_results.csv"))
    end
end
