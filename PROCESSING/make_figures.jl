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
    linecolor = (0.0,0.0,0.0)
    linecolor2 = (0.0,0.0,0.0)
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
    xbounds = (0.2,1)
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
        p = plot(xpts,ypts, label = "SH", color = RGBA(linecolor... ,1.0), xlims = xbounds,ylims = ybounds, legend = :topleft, xlabel="Radial Distance From Hub (r/R)", ylabel="Tangential Distance From Leading Edge (d/R)")
        plot!(Shape(shapex,shapey);color = RGBA(linecolor... ,0.35),linecolor = RGBA(0,0,0,0), label = "SH ± 1σ")
        plot!(xpts,ypts2, label = "Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0),linestyle = :dash)
        plot!(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), label = "Smooth Hydrophillic ± 1σ")
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = RGBA(0.25,0.25,0.25,1.0), label = "")#Propeller Hub
        return p
    else
        p = plot(xpts,ypts, label = "SH", color = RGBA(linecolor... ,1.0), xlims = xbounds,ylims = ybounds, legend = :topleft, xlabel="Radial Distance From Hub (r/R)", ylabel="Tangential distance From Leading Edge (d/R)")
        plot!(Shape(shapex,shapey);color = RGBA(linecolor... ,0.35),linecolor = RGBA(0,0,0,0), label = "SH ± 1σ")
        plot!(xpts,ypts2, label = "Smooth Hydrophillic", color = RGBA(linecolor2... ,1.0),linestyle = :dash)
        plot!(Shape(shapex2,shapey2);color = RGBA(linecolor2... ,0.2),linecolor = RGBA(0,0,0,0), label = "Smooth Hydrophillic ± 1σ")
        plot!(Shape([xbounds[1],0.32,0.32,xbounds[1]],[ybounds[1],ybounds[1],ybounds[2],ybounds[2]]),color = RGBA(0.25,0.25,0.25,1.0), label = "")
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

function makespectrumplottable(frequencies,spectrum1,spectrum2)
    x = []
    y1 = []
    y2 = []
    for j = 1:1:(length(frequencies)-1)
        push!(x,frequencies[j])
        push!(y1,spectrum1[j])
        push!(y2,spectrum2[j])
        push!(x,frequencies[j+1])
        push!(y1,spectrum1[j])
        push!(y2,spectrum2[j])
    end
    x[1] = 1.0
    return x, y1, y2
end

function getspectrumaroundspeed(speed,speeds,spectra; speedrange = 100.0)
    totalweight = 0.0
    spectrum = zeros(size(spectra)[2])
    for i = 1:1:length(speeds)
        if (speeds[i] < (speed + speedrange)) && (speeds[i] > (speed - speedrange))
            weight = 1.0 #This will be different if using a gaussian average
            totalweight = weight + totalweight
            spectrum  = spectrum + (weight*spectra[i,:])
        end
    end
    spectrum = spectrum ./ totalweight
    return spectrum
end

function analyzeaudio()
    props = ["1" "2" "3" "4" "5" "6" "7" "8"]
    slow = ["2" "6"]
    speeds_to_graph = [4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000 13500 14000 14500 15000]
    println("Loading control audio 1 / 2")
    temp1, normalizationspeedsslow, normalizationtorquesslow, temp2, normalizationspectraslow, hydrophone_avg_slow = getpropaudiodata("0_1")
    println("Loading control audio 2 / 2")
    temp1, normalizationspeedsfast, normalizationtorquesfast, temp2, normalizationspectrafast, hydrophone_avg_fast = getpropaudiodata("0_2")
    println("GPR fitting control audio 1 / 2")
    slownormalizationtorquefromspeed, slownormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsslow),Float64.(normalizationtorquesslow),Float64.(normalizationspectraslow))#TODO: get the slownormalizationtorquefromspeed from somewhere other than GPR
    println("GPR fitting control audio 2 / 2")
    fastnormalizationtorquefromspeed, fastnormalizationspectrumfromspeed = fittorqueandspectrumtospeed(Float64.(normalizationspeedsfast),Float64.(normalizationtorquesfast),Float64.(normalizationspectrafast))
    spectrafromspeeds = []
    torqueconstantssets = []
    cavitationnumberssets = []
    spectrasets = []
    frequencies = []
    speedsets = []
    hydrophone_avgs = []
    for i = 1:1:length(props)
        println("Reading propeller audio "*props[i]*" / $(length(props))")
        ts, speeds, torques, frequencies, spectra, hydrophone_avg = getpropaudiodata(props[i])
        push!(hydrophone_avgs,hydrophone_avg)
        push!(cavitationnumberssets,cavitationnumberfromRPM.(speeds;prop = props[i]))
        torqueconstants = similar(cavitationnumberssets[end])
        if contains(slow,props[i])
            push!(torqueconstantssets,torqueconstantsfromtorques(torques,speeds,slownormalizationtorquefromspeed))
        else
            push!(torqueconstantssets,torqueconstantsfromtorques(torques,speeds,fastnormalizationtorquefromspeed))
        end
        push!(spectrasets,spectra)
        push!(speedsets,speeds)
        #torquefromspeed, spectrumfromspeed = fittorqueandspectrumtospeed(Float64.(speeds),Float64.(torques),Float64.(spectra))
        #p =plot(spectra[100:150,:]; xscale = :log10)
        #savefig(p,"DATA/output/figures/test_$i"*".png")
        #push!(spectrafromspeeds,spectrumfromspeed)
        #prop = parse(Int64, props[i])
        #println(length(frequencies))
        #println(size(spectra))
    end
    #MAKE PLOTS----------------------
    nplots = 4 + 4*length(speeds_to_graph)
    nplotted = 0
    #torque vs cavitation
    for i = 1:1:4
        nplotted = nplotted + 1
        println("Plotting $nplotted / $nplots")
        p =plot(cavitationnumberssets[i],torqueconstantssets[i]; xscale = :log10)#TODO: plot formatting, labels, etc
        plot!(cavitationnumberssets[i+4],torqueconstantssets[i+4])
        savefig(p,"DATA/output/figures/torque_v_cavitation_$i"*".png")
    end
    for speed in speeds_to_graph
        slow_background_spectrum = getspectrumaroundspeed(speed,normalizationspeedsslow,normalizationspectraslow)
        fast_background_spectrum = getspectrumaroundspeed(speed,normalizationspeedsslow,normalizationspectraslow)
        xc, ycslow, ycfast = makespectrumplottable(frequencies,slow_background_spectrum,fast_background_spectrum)
        for i = 1:1:4
            nplotted = nplotted + 1
            println("Plotting $nplotted / $nplots")
            spectrum1 = getspectrumaroundspeed(speed,speedsets[i],spectrasets[i])
            spectrum2 = getspectrumaroundspeed(speed,speedsets[i+4],spectrasets[i+4])
            x, y1, y2 = makespectrumplottable(frequencies,spectrum1,spectrum2)
            p=plot(x,y2;xscale = :log10, label = "SH", color = RGBA(0,0,0,1), legend = :outertopright)#TODO: label with cavitation number
            plot!(x,y1; label = "Hydrophillic", linestyle = :dash, color = RGBA(0,0,0,1))
            if i == 2
                shift = 10*log10(hydrophone_avg_slow/(0.5*hydrophone_avgs[i] + 0.5*hydrophone_avgs[i+4]))#adjusts relative amplitude of control data
                #println(shift)
                #plot!(xc,ycslow .+ shift; label = "No Propeller")
            else
                shift = 10*log10(hydrophone_avg_fast/(0.5*hydrophone_avgs[i] + 0.5*hydrophone_avgs[i+4]))#adjusts relative amplitude of control data
                #println(shift)
                #plot!(xc,ycfast .+ shift; label = "No Propeller")
            end
            xlabel!("Frequency (Hz)")
            ylabel!("dB/Hz")
            savefig(p,"DATA/output/figures/PSD_$i"*"_$speed"*".png")
        end
    end
end
