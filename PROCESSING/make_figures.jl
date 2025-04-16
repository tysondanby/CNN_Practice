#-----ENVIRONMENT SETUP-------------------------------------------------------
using Pkg
Pkg.activate(@__DIR__) 
required_packages = ["Plots", "CSV", "DataFrames", "FFTW", "WAV", "GaussianProcesses"]
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
println(@__DIR__)
working_dir = dirname(@__DIR__)
cd(working_dir)
#-----END ENVIRONMENT SETUP---------------------------------------------------

using Plots, CSV, DataFrames, FFTW, WAV, Base.Threads, GaussianProcesses
include(pwd()*"/LIB/arrayoperations.jl")
include(pwd()*"/LIB/math.jl")
include(pwd()*"/LIB/macros.jl")
#include(pwd()*"/LIB/imageprocessing.jl")
threebladed = [4 8]
invert = [1 5]
imagedirectory = "DATA/camera/prepared"
manual_resultscsv = CSV.File("DATA/camera/model_data/manual_results.csv")

include(pwd()*"/PROCESSING/read_image_data_from_csv.jl")

#TODO: verify speed better for images. (perhaps do this manually?)
#=
function plotwitherror(xpts,ypts,stdevpts)
    linecolor = (1.0,0.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "")
    plot!(xpts,ypts, label = "Cavitation Bounds", color = RGBA(linecolor... ,1.0))
    plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
    index1 = findfirst(x->(x> 1.0),ypts)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if rtip < 1.0
        plot!(Shape([rtip,xbounds[2],xbounds[2],rtip],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = tipvortexcolor, label = "Tip Vortex Cavitation")
    end
    return p
end =#

function plotwitherrorboth(xpts,ypts,stdevpts,ypts2,stdevpts2)#TODO: edit this to have titles and legend in the correct spot
    linecolor = (1.0,0.0,0.0)
    linecolor2 = (0.0,1.0,0.0)
    tipvortexcolor = RGBA(0.6,0.7,1.0,1.0)
    tipvortexcolor2 = RGBA(0.6,0.7,1.0,1.0)
    yptshigh = ypts + stdevpts
    yptslow = ypts - stdevpts
    shapex = vcat(xpts,reverse(xpts))
    shapey = vcat(yptshigh,reverse(yptslow))
    yptshigh2 = ypts2 + stdevpts2
    yptslow2 = ypts2 - stdevpts2
    shapex2 = vcat(xpts,reverse(xpts))
    shapey2 = vcat(yptshigh2,reverse(yptslow2))
    xbounds = (0,1)
    ybounds = (-0.05,1)
    
    index1 = findfirst(x->(x> 1.0),ypts)
    index2 = findfirst(x->(x> 1.0),ypts2)
    if index1 == nothing
        rtip = 1.1
    else
        rtip = xpts[index1]
    end
    if index2 == nothing
        rtip2 = 1.1
    else
        rtip2 = xpts[index2]
    end
    if rtip < rtip2
        p = plot(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        plot!(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    else
        p = plot(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), xlims = xbounds,ylims = ybounds, label = "", legend = :topleft)
        plot!(xpts,ypts2, label = "Cavitation Bounds - Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0))
        plot!(Shape(shapex,shapey);color = RGBA(linecolor... ,0.2),linecolor = RGBA(0,0,0,0), label = "")
        plot!(xpts,ypts, label = "Cavitation Bounds - SH", color = RGBA(linecolor... ,1.0))
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = :gray, label = "Propeller Hub")
        return p
    end
end

function plotspeedboth(propnumber,speed)
    rclean, thout, stdev = getpropspeeddatafromcsv(propnumber,speed)
    rclean, thout2, stdev2 = getpropspeeddatafromcsv(propnumber+4,speed)
    return plotwitherrorboth(rclean, thout, stdev,thout2, stdev2)
end

function create_cavitation_location_figures()
    speeds = collect(4500:500:15000)
    props = collect(1:1:4)
    for prop in props
        for speed in speeds
            p = plotspeedboth(prop,speed)
            savefig(p,"DATA/output/figures/highspeed_$prop"*"_$speed"*".png")
        end
    end
end

#--------------AUDIO
include(pwd()*"/PROCESSING/prune_audio.jl")

include(pwd()*"/PROCESSING/read_wav.jl")

function getpressure(prop)
    if (prop == "5") || (prop == "7")
        return 85337.0, 3070.0 # 88000, 2800
    else
        return 85337.0, 3070.0
    end
end

include(pwd()*"/PROCESSING/nondimensionalize_audio.jl")

include(pwd()*"/PROCESSING/audio_GPR.jl")

function analyzeaudio()
    props = ["1" "2" "3" "4" "5" "6" "7" "8"]
    slow = ["2" "6"]
    speeds_to_graph = [4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000 13500 14000 14500 15000]
    temp1, normalizationspeedsslow, normalizationtorquesslow, temp2, normalizationspectraslow = getpropaudiodata("0_1")
    temp1, normalizationspeedsfast, normalizationtorquesfast, temp2, normalizationspectrafast = getpropaudiodata("0_2")
    slownormalizationtorquefromspeed, slownormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsslow),Float64.(normalizationtorquesslow),Float64.(normalizationspectraslow))
    fastnormalizationtorquefromspeed, fastnormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsfast),Float64.(normalizationtorquesfast),Float64.(normalizationspectrafast))
    spectrafromspeeds = []
    torqueconstantssets = []
    cavitationnumberssets = []
    spectrasets = []
    frequencies = []
    for i = 1:1:length(props)
        println(props[i])
        ts, speeds, torques, frequencies, spectra = getpropaudiodata(props[i])
        push!(cavitationnumberssets,cavitationnumberfromRPM.(speeds;prop = props[i]))
        torqueconstants = similar(cavitationnumberssets[end])
        if contains(slow,props[i])
            push!(torqueconstantssets,torqueconstantsfromtorques(torques,speeds,slownormalizationtorquefromspeed))
        else
            push!(torqueconstantssets,torqueconstantsfromtorques(torques,speeds,fastnormalizationtorquefromspeed))
        end
        push1(spectrasets,spectra)
        torquefromspeed, spectrumfromspeed = fittorqueandspectrumtospeed(Float64.(speeds),Float64.(torques),Float64.(spectra))
        p =plot(spectra[100:150,:]; xscale = :log10)
        savefig(p,"DATA/output/figures/test_$i"*".png")
        push!(spectrafromspeeds,spectrumfromspeed)
        prop = parse(Int64, props[i])
        println(length(frequencies))
        println(size(spectra))
    end
    #MAKE PLOTS----------------------
    #torque vs cavitation
    for i = 1:1:4
        p =plot(cavitationnumberssets[i],torqueconstantssets[i]; xscale = :log10)#TODO: plot formatting, labels, etc
        plot!(cavitationnumberssets[i+4],torqueconstantssets[i+4])
        savefig(p,"DATA/output/figures/torque_v_cavitation_$i"*".png")
        #spectra at each speed
        #TODO: Stop using GPR
        for speed in speeds_to_graph #TODO: label with cavitation number. Convert to PSD
            xoriginal = frequencies
            y1original = spectrafromspeeds[i](speed)#TODO: subtract background
            y2original = spectrafromspeeds[i+4](speed)
            println(y1original)
            x = []
            y1 = []
            y2 = []
            for j = 1:1:(length(frequencies)-1)
                push!(x,xoriginal[j])
                push!(y1,y1original[j])
                push!(y2,y2original[j])
                push!(x,xoriginal[j+1])
                push!(y1,y1original[j])
                push!(y2,y2original[j])
            end
            p2 =plot(x,[y1 y2]; xscale = :log10, xlims = (1,100000))
            savefig(p2,"DATA/output/figures/PSD_$i"*"_$speed.png")
        end
    end
    

end
